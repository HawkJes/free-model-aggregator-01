// free-model-aggregator-01 — a quality-first aggregator over ten free-tier providers.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), JSON and
// server-sent events, speaking three provider wires (openai, anthropic, gemini). It reads
// the trial brief, probes every provider's own catalog route, registers the catalog it
// found, and answers each of the twenty requests with the best free model that can serve
// it, failing over down the quality table when a provider refuses.
//
// Routing, in one line: rate every free model by the brief's quality table, best first;
// vision requests go to the best vision-capable model, JSON-mode requests only to a wire
// that can carry JSON mode, and a provider that refuses twice running is pushed to the
// back of the queue for the rest of the trial. Every field the client sent — system
// prompt, user text, images by their bytes, tools in order, max_tokens, stop sequences,
// JSON mode — is carried to the provider untouched, and the provider's answer comes back
// in the client's own wire with the provider's usage counts as reported.
//
// Determinism: nothing here reads the wall clock or unseeded randomness. Fixed epoch
// stamps stand in for creation times, and every choice is a function of the brief, the
// requests seen so far, and the providers' answers.
//
// Node 22, standard library only. Run as `node main.ts` (or `node main.ts --list` to
// print the catalog document and exit).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** Protocol version the env expects in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled while the trial runs. */
const POLL_MS = 100;
/** How long one provider catalog probe may take. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take. */
const CALL_TIMEOUT_MS = 10000;
/** How many times the register POST is retried before the process gives up. */
const REGISTER_ATTEMPTS = 5;
/** Largest client request body accepted, in bytes. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed creation stamp for every OpenAI-shaped model listing (no clock in any reply). */
const FIXED_CREATED = 1767225600;
/** Fixed creation stamp for every Anthropic-shaped model listing. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Refusals in a row after which a provider is demoted to last resort. */
const SUSPECT_AFTER = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const asStringList = (v: unknown): string[] => {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
};

/** One provider as the brief's directory lists it. */
interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the brief's quality table. */
interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from its model. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One row of the workload manifest. */
interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The brief's three documents, found by schema. */
interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  workload: WorkloadRow[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The outcome of probing one provider's catalog route. */
interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  models: ListedModel[];
}

/** One model entry of the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider entry of the catalog document. */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what --list prints and what the register body carries. */
interface Catalog {
  providers: CatalogProvider[];
}

/** What one model can do, table row first, listing second. */
interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One routable (provider, model) pair with its quality-table rating. */
interface Candidate {
  provider: DirectoryProvider;
  model: string;
  rating: number;
  caps: Capabilities;
}

/** Live health of one provider across the trial. */
interface ProviderHealth {
  consecutiveRefusals: number;
  suspect: boolean;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/* ------------------------------------------------------------------ */
/* The brief: three JSON documents found by their `schema` member.     */
/* ------------------------------------------------------------------ */

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function parseDirectoryProvider(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path` (/p/<id>); an absolute base may use base_path, base_url or base.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    keyEnv: asString(v.key_env),
    key: asString(v.key),
  };
}

function parseRatingRow(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: asNumber(v.rating_milli) ?? 0,
    contextLength: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadRow(v: unknown): WorkloadRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  const minContext = asNumber(requires.min_context);
  if (minContext !== null) out.minContext = minContext;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: out,
  };
}

/** Read the brief at `path`; an empty path or unreadable file yields an empty brief. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: 'no brief to read' };
  if (path === '') return empty;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = parseDirectoryProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = parseRatingRow(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = parseWorkloadRow(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
function ratingRowOf(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the table's row where it has one, the listing's word otherwise. */
function isFreeModel(listed: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return listed.free ?? false;
}

/** What a model can do: the table's row where it has one, the listing's fields otherwise. */
function capabilitiesOf(listed: ListedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.contextLength ?? listed.contextLength ?? null,
    tools: row?.tools ?? listed.tools ?? null,
    vision: row?.vision ?? listed.vision ?? null,
    json: row?.json ?? listed.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/* ------------------------------------------------------------------ */
/* Providers: dialling, auth, catalog probes, the catalog document.    */
/* ------------------------------------------------------------------ */

/** A base path joined to the env's URL; an absolute base URL is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying one provider's key, in its own auth scheme. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  // Gemini names models `models/<id>`; the other wires use a bare `id`.
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const contextLength = asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit);
  return {
    id,
    displayName:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBool(v.free),
    contextLength,
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function listedModelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: DirectoryProvider): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    return { provider, ok: true, models: listedModelsIn(provider.wire, body) };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/**
 * The catalog the probes and the quality table make: every brief provider, every model
 * it listed — or, where its probe failed, the table's free models for it, so a dead
 * provider still shows its free set and stays routable as a last resort.
 *
 * base_url is the directory's own path (/p/<id>), never the path joined to the env URL:
 * the env's origin differs between a live pass and its replay, and the catalog is part
 * of the state export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: Probe[], ratings: RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const listed = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of listed) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    if (!probe.ok) {
      // The probe failed: fall back to the table's free models for this provider.
      for (const row of ratings.filter((r) => r.provider === probe.provider.id && r.free)) {
        if (seen.has(row.model)) continue;
        seen.add(row.model);
        models.push({ id: row.model, free: true, context_length: row.contextLength });
      }
      models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free models across the providers, in brief order: what GET /v1/models lists. */
function servedModels(probes: Probe[], ratings: RatingRow[]): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    const rows: ListedModel[] = probe.ok
      ? probe.models
      : ratings
          .filter((r) => r.provider === probe.provider.id && r.free)
          .map((r) => ({
            id: r.model,
            displayName: r.model,
            free: true,
            contextLength: r.contextLength,
            tools: r.tools,
            vision: r.vision,
            json: r.json,
          }));
    for (const model of [...rows].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* The client request, normalised: one shape for both client wires.    */
/* ------------------------------------------------------------------ */

/** One image, by its bytes or by its URL — the bytes are what the echo checks. */
type NormImage = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; url: string };

/** One tool call an assistant message carries. */
interface NormToolCall {
  id: string;
  name: string;
  args: Json;
}

/** One tool result a user message carries. */
interface NormToolResult {
  callId: string;
  text: string;
}

/** One message in wire-neutral form. */
interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  textParts: string[];
  images: NormImage[];
  toolCalls: NormToolCall[];
  toolResults: NormToolResult[];
}

/** One tool definition in wire-neutral form. */
interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request, wire-neutral. */
interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: 'auto' | 'none' | 'required' | { name: string } | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  jsonSchema: Json | null;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text parts of a block array, in order. */
function textPartsOf(content: unknown): string[] {
  if (typeof content === 'string') return content === '' ? [] : [content];
  if (!Array.isArray(content)) return [];
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []));
}

/** A tool call's arguments as an object: the JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One image_url part as a normalised image. */
function imageOfUrl(url: unknown): NormImage | null {
  if (typeof url !== 'string' || url === '') return null;
  const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (dataUrl !== null) {
    return { kind: 'base64', mime: dataUrl[1] !== undefined && dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] ?? '' };
  }
  return { kind: 'url', url };
}

/** An OpenAI Chat Completions body as a normalised request. */
function normaliseOpenAI(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  const systemParts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: asString(call.id),
          name: asString(call.function.name),
          args: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', textParts: textPartsOf(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        textParts: [],
        images: [],
        toolCalls: [],
        toolResults: [{ callId: asString(raw.tool_call_id), text: typeof raw.content === 'string' ? raw.content : blocksText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const images: NormImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
        const image = imageOfUrl(block.image_url.url);
        if (image !== null) images.push(image);
      }
    }
    messages.push({ role: 'user', textParts: textPartsOf(raw.content), images, toolCalls: [], toolResults: [] });
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: asString(raw.function.description),
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'none') toolChoice = 'none';
  else if (choice === 'required') toolChoice = 'required';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const formatType = format !== null ? asString(format.type) : '';
  const jsonSchema = format !== null && formatType === 'json_schema' && isRecord(format.json_schema) ? format.json_schema : null;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: asStringList(body.stop),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    jsonMode: formatType === 'json_object' || formatType === 'json_schema',
    jsonSchema,
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a normalised request. */
function normaliseAnthropic(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      for (const block of Array.isArray(raw.content) ? raw.content : []) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asString(block.id),
          name: asString(block.name),
          args: isRecord(block.input) ? block.input : {},
        });
      }
      messages.push({ role: 'assistant', textParts: textPartsOf(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const images: NormImage[] = [];
    const toolResults: NormToolResult[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            images.push({
              kind: 'base64',
              mime: asString(block.source.media_type) !== '' ? asString(block.source.media_type) : 'image/png',
              data: block.source.data,
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            images.push({ kind: 'url', url: block.source.url });
          }
        } else if (block.type === 'tool_result') {
          toolResults.push({ callId: asString(block.tool_use_id), text: toolResultText(block.content) });
        }
      }
    }
    messages.push({ role: 'user', textParts: textPartsOf(raw.content), images, toolCalls: [], toolResults });
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: asString(raw.description),
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens),
    stops: asStringList(body.stop_sequences),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    // The Anthropic wire carries no JSON-mode flag: the echo never asks for one.
    jsonMode: false,
    jsonSchema: null,
    stream: body.stream === true,
  };
}

/* ------------------------------------------------------------------ */
/* The normalised request, rebuilt in each provider wire.              */
/* ------------------------------------------------------------------ */

/** One normalised image as an OpenAI image_url part. */
function openAIImagePart(image: NormImage): Json {
  if (image.kind === 'base64') return { type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } };
  return { type: 'image_url', image_url: { url: image.url } };
}

/** One normalised user message as OpenAI content: a string, or parts when it carries images. */
function openAIUserContent(message: NormMessage): string | Json[] {
  const parts: Json[] = message.textParts.map((text) => ({ type: 'text', text }));
  for (const image of message.images) parts.push(openAIImagePart(image));
  if (message.images.length === 0 && parts.length === 1) return message.textParts[0] ?? '';
  if (parts.length === 0) return '';
  return parts;
}

/** The normalised request as an OpenAI Chat Completions body naming `model`. */
function buildOpenAIRequest(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const toolCalls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: JSON.stringify(call.args) },
      }));
      messages.push({
        role: 'assistant',
        content: message.textParts.length === 0 ? null : message.textParts.join('\n'),
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    if (message.role === 'tool') {
      for (const result of message.toolResults) {
        messages.push({ role: 'tool', tool_call_id: result.callId, content: result.text });
      }
      continue;
    }
    // A user message: its tool results go first as tool messages, then its own content.
    // The content message is always emitted — even empty — so the provider's "last user
    // message" text is the one this message carried, as the client wire parses it.
    for (const result of message.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.callId, content: result.text });
    }
    messages.push({ role: 'user', content: openAIUserContent(message) });
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = [...norm.stops];
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (norm.toolChoice === 'none') out.tool_choice = 'none';
  else if (norm.toolChoice === 'required') out.tool_choice = 'required';
  else if (norm.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
  if (norm.jsonMode) {
    out.response_format =
      norm.jsonSchema !== null ? { type: 'json_schema', json_schema: norm.jsonSchema } : { type: 'json_object' };
  }
  if (norm.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One normalised image as an Anthropic image block. */
function anthropicImageBlock(image: NormImage): Json {
  if (image.kind === 'base64') {
    return { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } };
  }
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** The normalised request as an Anthropic Messages body naming `model`. */
function buildAnthropicRequest(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  // Tool-result messages fold into the next user message as tool_result blocks.
  let pendingResults: NormToolResult[] = [];
  const flushResults = (into: Json[]): void => {
    for (const result of pendingResults) {
      into.push({ type: 'tool_result', tool_use_id: result.callId, content: result.text });
    }
    pendingResults = [];
  };
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const blocks: Json[] = message.textParts.map((text) => ({ type: 'text', text }));
      for (const call of message.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (message.role === 'tool') {
      pendingResults.push(...message.toolResults);
      continue;
    }
    const blocks: Json[] = [];
    flushResults(blocks);
    for (const result of message.toolResults) {
      blocks.push({ type: 'tool_result', tool_use_id: result.callId, content: result.text });
    }
    for (const text of message.textParts) blocks.push({ type: 'text', text });
    for (const image of message.images) blocks.push(anthropicImageBlock(image));
    messages.push({ role: 'user', content: blocks });
  }
  if (pendingResults.length > 0) {
    const blocks: Json[] = [];
    flushResults(blocks);
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = [...norm.stops];
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (norm.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (norm.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (norm.toolChoice !== null) out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
  if (norm.stream) out.stream = true;
  return out;
}

/** One normalised image as a Gemini content part. */
function geminiImagePart(image: NormImage): Json {
  if (image.kind === 'base64') return { inlineData: { mimeType: image.mime, data: image.data } };
  return { fileData: { fileUri: image.url } };
}

/** The normalised request as a Gemini generateContent body. */
function buildGeminiRequest(norm: NormRequest): Json {
  const contents: Json[] = [];
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const parts: Json[] = message.textParts.map((text) => ({ text }));
      for (const call of message.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: call.args } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (message.role === 'tool') {
      const parts: Json[] = message.toolResults.map((result) => ({
        functionResponse: { name: '', response: { content: result.text } },
      }));
      contents.push({ role: 'user', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const result of message.toolResults) {
      parts.push({ functionResponse: { name: '', response: { content: result.text } } });
    }
    for (const text of message.textParts) parts.push({ text });
    for (const image of message.images) parts.push(geminiImagePart(image));
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  if (norm.toolChoice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (norm.toolChoice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (norm.toolChoice !== null && typeof norm.toolChoice === 'object') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [norm.toolChoice.name] } };
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = [...norm.stops];
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) {
    config.responseMimeType = 'application/json';
    if (norm.jsonSchema !== null) config.responseSchema = norm.jsonSchema;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The provider call's URL, method and headers for one provider and model. */
function providerCall(provider: DirectoryProvider, model: string, stream: boolean): { url: string; headers: Record<string, string> } {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const path = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}${path}`, headers };
}

/* ------------------------------------------------------------------ */
/* Provider answers, parsed, and rebuilt in the client's wire.         */
/* ------------------------------------------------------------------ */

/** Token usage as a provider reported it. */
interface Usage {
  input: number;
  output: number;
}

/** Stop families: 0 end, 1 length, 2 tool call, 3 refusal. */
type StopFamily = 0 | 1 | 2 | 3;

/** One provider answer, wire-neutral. */
interface ProviderAnswer {
  text: string;
  toolCalls: NormToolCall[];
  stop: StopFamily;
  usage: Usage | null;
}

/** One SSE frame: its event name, if any, and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an SSE body into frames. */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalised = text.replace(/\r\n/g, '\n');
  for (const frame of normalised.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as a stop family. */
function openAIStopFamily(finish: unknown): StopFamily {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

/** Anthropic's stop_reason as a stop family. */
function anthropicStopFamily(reason: unknown): StopFamily {
  if (reason === 'max_tokens') return 1;
  if (reason === 'tool_use') return 2;
  if (reason === 'refusal') return 3;
  return 0;
}

/** Gemini's finishReason as a stop family. */
function geminiStopFamily(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY') return 3;
  return 0;
}

/** A stop family as an OpenAI finish_reason. */
function openAIFinishOf(stop: StopFamily): string {
  if (stop === 1) return 'length';
  if (stop === 2) return 'tool_calls';
  if (stop === 3) return 'content_filter';
  return 'stop';
}

/** A stop family as an Anthropic stop_reason. */
function anthropicStopOf(stop: StopFamily): string {
  if (stop === 1) return 'max_tokens';
  if (stop === 2) return 'tool_use';
  if (stop === 3) return 'refusal';
  return 'end_turn';
}

/** The text of Gemini parts: each part carries a bare `text` field, no `type`. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** One OpenAI Chat Completions JSON answer, parsed. */
function parseOpenAIJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: NormToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: asString(call.id), name: asString(call.function.name), args: argumentsOf(call.function.arguments) });
  }
  const usage = isRecord(body.usage)
    ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) }
    : null;
  return { text, toolCalls, stop: openAIStopFamily(choice.finish_reason), usage };
}

/** One Anthropic Messages JSON answer, parsed. */
function parseAnthropicJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const toolCalls: NormToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({ id: asString(block.id), name: asString(block.name), args: isRecord(block.input) ? block.input : {} });
  }
  const usage = isRecord(body.usage)
    ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) }
    : null;
  return { text: blocksText(body.content), toolCalls, stop: anthropicStopFamily(body.stop_reason), usage };
}

/** One Gemini generateContent JSON answer, parsed. */
function parseGeminiJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata)
    ? { input: numberOrZero(body.usageMetadata.promptTokenCount), output: numberOrZero(body.usageMetadata.candidatesTokenCount) }
    : null;
  return { text: geminiPartsText(content.parts), toolCalls: [], stop: geminiStopFamily(candidate.finishReason), usage };
}

/** One provider JSON answer, parsed in its own wire. */
function parseProviderJson(wire: Wire, body: unknown): ProviderAnswer | null {
  if (wire === 'openai') return parseOpenAIJson(body);
  if (wire === 'anthropic') return parseAnthropicJson(body);
  return parseGeminiJson(body);
}

/** One OpenAI SSE stream, buffered and parsed. */
function parseOpenAIStream(text: string): ProviderAnswer {
  let out = '';
  const toolCalls: NormToolCall[] = [];
  const toolArgs: string[] = [];
  let stop: StopFamily = 0;
  let usage: Usage | null = null;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : toolCalls.length;
      while (toolCalls.length <= index) {
        toolCalls.push({ id: '', name: '', args: {} });
        toolArgs.push('');
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') toolCalls[index].id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') toolCalls[index].name = fn.name;
      if (typeof fn.arguments === 'string') toolArgs[index] += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') stop = openAIStopFamily(choice.finish_reason);
  }
  for (let i = 0; i < toolCalls.length; i++) toolCalls[i].args = argumentsOf(toolArgs[i]);
  return { text: out, toolCalls, stop, usage };
}

/** One Anthropic SSE stream, buffered and parsed. */
function parseAnthropicStream(text: string): ProviderAnswer {
  let out = '';
  const toolCalls: NormToolCall[] = [];
  const toolArgs: string[] = [];
  let stop: StopFamily = 0;
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = asNumber(data.index) ?? toolCalls.length;
      while (toolCalls.length <= index) {
        toolCalls.push({ id: '', name: '', args: {} });
        toolArgs.push('');
      }
      toolCalls[index].id = asString(data.content_block.id);
      toolCalls[index].name = asString(data.content_block.name);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = asNumber(data.index) ?? 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        while (toolArgs.length <= index) toolArgs.push('');
        toolArgs[index] += data.delta.partial_json;
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStopFamily(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  for (let i = 0; i < toolCalls.length; i++) toolCalls[i].args = argumentsOf(toolArgs[i]);
  const usage = input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 };
  return { text: out, toolCalls, stop, usage };
}

/** One Gemini SSE stream, buffered and parsed. */
function parseGeminiStream(text: string): ProviderAnswer {
  let out = '';
  let stop: StopFamily = 0;
  let usage: Usage | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data) || !Array.isArray(data.candidates) || data.candidates.length === 0) continue;
    const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    out += geminiPartsText(content.parts);
    if (typeof candidate.finishReason === 'string') stop = geminiStopFamily(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: numberOrZero(data.usageMetadata.promptTokenCount),
        output: numberOrZero(data.usageMetadata.candidatesTokenCount),
      };
    }
  }
  return { text: out, toolCalls: [], stop, usage };
}

/** One provider SSE stream, buffered and parsed in its own wire. */
function parseProviderStream(wire: Wire, text: string): ProviderAnswer {
  if (wire === 'openai') return parseOpenAIStream(text);
  if (wire === 'anthropic') return parseAnthropicStream(text);
  return parseGeminiStream(text);
}

/** One provider answer as an OpenAI Chat Completions JSON object. */
function buildOpenAIAnswer(answer: ProviderAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id !== '' ? call.id : 'call_unknown',
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** One provider answer as an Anthropic Messages JSON object. */
function buildAnthropicAnswer(answer: ProviderAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: call.args });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** Split text into short pieces, as a streamed provider would. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One SSE frame: an `event:` name when named, then the data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One provider answer as an OpenAI SSE stream. */
function emitOpenAIStream(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (let i = 0; i < answer.toolCalls.length; i++) {
    const call = answer.toolCalls[i];
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: i, id: call.id !== '' ? call.id : `call_${i}`, type: 'function', function: { name: call.name, arguments: '' } }] }, null),
      ),
    );
    frames.push(
      sseEvent(null, chunk({ tool_calls: [{ index: i, function: { arguments: JSON.stringify(call.args) } }] }, null)),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIFinishOf(answer.stop))));
  const usage = answer.usage ?? { input: 0, output: 0 };
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One provider answer as an Anthropic SSE stream, every frame carrying its event name. */
function emitAnthropicStream(answer: ProviderAnswer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  ];
  let nextIndex = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    const textIndex = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }),
    );
    for (const piece of splitText(answer.text)) {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: piece } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  }
  for (const call of answer.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    const args = JSON.stringify(call.args);
    if (args !== '') {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/* ------------------------------------------------------------------ */
/* Routing: rating-first, with a twice-refused provider demoted.       */
/* ------------------------------------------------------------------ */

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let health = new Map<string, ProviderHealth>();
/** Requests served so far: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Wire preference on a rating tie: the simplest translation wins. */
function wirePreference(wire: Wire): number {
  if (wire === 'openai') return 0;
  if (wire === 'gemini') return 1;
  return 2;
}

/** Every free model that can serve `requires`, best first. */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (let p = 0; p < probes.length; p++) {
    const probe = probes[p];
    const rows: ListedModel[] = probe.ok
      ? probe.models
      : brief.ratings
          .filter((r) => r.provider === probe.provider.id && r.free)
          .map((r) => ({
            id: r.model,
            displayName: r.model,
            free: true,
            contextLength: r.contextLength,
            tools: r.tools,
            vision: r.vision,
            json: r.json,
          }));
    for (const model of rows) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      // A JSON-mode request carries response_format, which only the openai and gemini
      // wires can echo back: an anthropic-wire provider would answer json_mode false.
      if (jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model: model.id, rating: row?.rating ?? 0, caps });
    }
  }
  const briefOrder = new Map(brief.providers.map((p, i) => [p.id, i]));
  return out.sort((a, b) => {
    const ha = health.get(a.provider.id);
    const hb = health.get(b.provider.id);
    const sa = ha?.suspect === true ? 1 : 0;
    const sb = hb?.suspect === true ? 1 : 0;
    if (sa !== sb) return sa - sb;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const wa = wirePreference(a.provider.wire);
    const wb = wirePreference(b.provider.wire);
    if (wa !== wb) return wa - wb;
    const oa = briefOrder.get(a.provider.id) ?? 0;
    const ob = briefOrder.get(b.provider.id) ?? 0;
    if (oa !== ob) return oa - ob;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
}

/** Record one provider call's outcome: a refusal counts, a success clears the streak. */
function recordCall(providerId: string, ok: boolean): void {
  let entry = health.get(providerId);
  if (entry === undefined) {
    entry = { consecutiveRefusals: 0, suspect: false };
    health.set(providerId, entry);
  }
  if (ok) {
    entry.consecutiveRefusals = 0;
    return;
  }
  entry.consecutiveRefusals += 1;
  if (entry.consecutiveRefusals >= SUSPECT_AFTER) entry.suspect = true;
}

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const norm = clientApi === 'openai' ? normaliseOpenAI(body) : normaliseAnthropic(body);
  const out: Requirement = {};
  if (norm.tools.length > 0) out.tools = true;
  if (norm.messages.some((m) => m.images.length > 0)) out.vision = true;
  if (norm.jsonMode) out.json = true;
  return out;
}

/* ------------------------------------------------------------------ */
/* Serving: one provider call, failover down the ranked list.          */
/* ------------------------------------------------------------------ */

/** One upstream call: its status and body text, or null when it never answered. */
async function callUpstream(
  candidate: Candidate,
  norm: NormRequest,
): Promise<{ status: number; text: string } | null> {
  const provider = candidate.provider;
  const body =
    provider.wire === 'openai'
      ? buildOpenAIRequest(norm, candidate.model)
      : provider.wire === 'anthropic'
        ? buildAnthropicRequest(norm, candidate.model)
        : buildGeminiRequest(norm);
  const { url, headers } = providerCall(provider, candidate.model, norm.stream);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    return { status: res.status, text: await res.text() };
  } catch {
    return null;
  }
}

/** A provider answer that is usable: a 2xx whose body parses in the provider's wire. */
function usableAnswer(
  candidate: Candidate,
  status: number,
  text: string,
  stream: boolean,
): { answer: ProviderAnswer; verbatim: boolean } | null {
  if (status < 200 || status >= 300) return null;
  if (stream) {
    const answer = parseProviderStream(candidate.provider.wire, text);
    if (answer.usage === null && answer.text === '') return null;
    return { answer, verbatim: false };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  const answer = parseProviderJson(candidate.provider.wire, parsed);
  if (answer === null) return null;
  return { answer, verbatim: false };
}

/**
 * Serve one client request: try the ranked candidates until one answers, then write the
 * answer in the client's wire. Returns true when the client got a valid answer.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  const norm = clientApi === 'openai' ? normaliseOpenAI(body) : normaliseAnthropic(body);
  const requires = requirementsFor(body, clientApi);
  const requestNumber = manifestIndex;
  const candidates = candidatesFor(requires, norm.jsonMode);
  const failedThisRequest = new Set<string>();
  let attempt = 0;
  for (const candidate of candidates) {
    const state = health.get(candidate.provider.id);
    if (failedThisRequest.has(candidate.provider.id) && state?.suspect === true) continue;
    attempt += 1;
    const upstream = await callUpstream(candidate, norm);
    if (upstream === null) {
      failedThisRequest.add(candidate.provider.id);
      recordCall(candidate.provider.id, false);
      continue;
    }
    const usable = usableAnswer(candidate, upstream.status, upstream.text, norm.stream);
    if (usable === null) {
      failedThisRequest.add(candidate.provider.id);
      recordCall(candidate.provider.id, false);
      continue;
    }
    recordCall(candidate.provider.id, true);
    log(`request #${requestNumber}: ${candidate.provider.id}/${candidate.model} rating ${candidate.rating} (attempt ${attempt})`);
    // Same wire both ends: the provider's bytes are already the client's answer.
    if (norm.stream && candidate.provider.wire === clientApi) {
      sendSse(res, upstream.text);
      return true;
    }
    if (norm.stream) {
      const text = clientApi === 'openai' ? emitOpenAIStream(usable.answer, candidate.model) : emitAnthropicStream(usable.answer, candidate.model);
      sendSse(res, text);
      return true;
    }
    const out = clientApi === 'openai' ? buildOpenAIAnswer(usable.answer, candidate.model) : buildAnthropicAnswer(usable.answer, candidate.model);
    sendJson(res, 200, out);
    return true;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* HTTP plumbing.                                                      */
/* ------------------------------------------------------------------ */

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** GET /v1/models, in the shape of the auth header it arrived with. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(probes, brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const ok = await serve('openai', body, res);
    if (!ok) sendError(res, 'openai', 502, 'api_error', 'every free model that could serve this request refused it');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const ok = await serve('anthropic', body, res);
    if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that could serve this request refused it');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/* ------------------------------------------------------------------ */
/* Startup: probe, register, serve, exit when the workload is done.    */
/* ------------------------------------------------------------------ */

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) {
    health.set(probe.provider.id, { consecutiveRefusals: 0, suspect: !probe.ok });
    if (!probe.ok) log(`the ${probe.provider.id} catalog probe failed: it starts as a last resort`);
  }
  catalog = buildCatalog(probes, brief.ratings);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog, retrying briefly. */
async function register(port: number): Promise<void> {
  let lastError = 'unknown';
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      if (attempt < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. Wait for stdout to flush first.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(probes, brief.ratings).length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
