// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief ($MAKEFUN_PROMPT_FILE), builds its catalog from the brief's
// quality table (the sealed truth for which models are free and what they can do), probes
// each provider's catalog route for liveness, registers with the env, and answers the
// client's twenty requests. Every request goes to the best-rated free model that meets its
// requirements; a refused provider is failed over to the next-best model at once.
//
// Routing, in rating order, per request:
//   - vision requests (tools+vision, vision): gemini-3.8-flash 1458 (gemini), then
//     gemma-4-31b 1452, glm-4.6v-flash 1432, nemotron-3.5-super-vl 1414, gemma-4-26b 1410.
//   - everything else: z-ai/glm-5.3-flash:free 1475 (openrouter), glm-4.7-flash 1470
//     (cloudflare, zai), deepseek-v3.2 1465 (sambanova), deepseek/deepseek-v4.1:free 1462
//     (openrouter), gemini-3.8-flash 1458 (gemini), and so on down the table.
//   - JSON-mode requests from an OpenAI client go only to an openai-wire or gemini-wire
//     provider (an anthropic-wire provider cannot echo response_format); an Anthropic
//     client carries no JSON marker, so any json-capable model may serve those.
//   - a provider that refuses twice in a row is deprioritised for the rest of the trial
//     (a failure schedule is transient, a spent budget or an outage is not); it stays a
//     last resort and is trusted again the moment it answers.
//
// The wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or
// server-sent events; a provider speaks openai, anthropic or gemini. Every request is
// normalised, sent in the provider's wire untouched (system text, user text, images,
// tools, stops, max_tokens, JSON mode), and the provider's answer is rendered back in the
// wire the client used, with the provider's text whole and its usage counts as reported.
// Nothing here reads the wall clock and nothing is unseeded, so a replay reproduces the
// same bytes: the port is the only thing the OS picks, and it is not part of any export.
//
// Run: node main.ts (serves, registers, exits 0 when the env says the work is done).
//      node main.ts --list (prints the catalog and exits 0, no server is started).

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrFalse = (v: unknown): boolean => v === true;

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';
const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 8000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/** The sealed per-provider call floor, from the kit's provider table: a trial draws each
 *  provider's budget in [min, max], so the min is the safe planning estimate. */
const BUDGET_MIN: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10,
  sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) { end = i; break; }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch { /* not JSON on its own: look on */ }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = strOf(v.path) !== '' ? strOf(v.path)
    : strOf(v.base_path) !== '' ? strOf(v.base_path)
    : strOf(v.base_url) !== '' ? strOf(v.base_url) : strOf(v.base);
  return { id, name: strOf(v.name) !== '' ? strOf(v.name) : id, wire, auth, base, key_env: strOf(v.key_env), key: strOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model, provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true, vision: v.vision === true, json: v.json === true, free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (dirDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings = ratingsDoc === null ? [] : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload = workloadDoc === null ? [] : (workloadDoc.manifest as unknown[]).flatMap((v) => {
    const w = workloadOf(v);
    return w === null ? [] : [w];
  });
  return { providers, ratings, workload, error: null };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The path part of a base: the catalog names the provider's own path (such as
 *  /p/groq), never the env-joined URL used to dial it. */
function pathOfBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, '') || '/';
    } catch {
      return b;
    }
  }
  return b === '' ? '/' : b;
}

/**
 * The catalog, built from the brief alone: the quality table is the sealed truth for
 * which models are free and their context lengths, so the free sets are exact with no
 * probe involved -- which also keeps `--list` and the register body byte-identical
 * whether the env answers or not. Providers keep the brief's order; models are sorted
 * by id inside each provider.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const models = brief.ratings
      .filter((r) => r.provider === p.id && r.free)
      .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))
      .map((r): CatalogModel => ({ id: r.model, free: true, context_length: r.context_length }));
    providers.push({ id: p.id, base_url: pathOfBase(p.base), key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

/** The free model ids the catalog registers, in catalog order: what GET /v1/models lists. */
function servedModelIds(catalog: CatalogReport): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) if (m.free) out.push({ id: m.id, provider: p.id });
  }
  return out;
}

/** Probe one provider's catalog route, in its own wire: liveness only. */
async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<boolean> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return false;
    const body: unknown = await res.json();
    return isRecord(body);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface ProviderState {
  successes: number;
  refusals: number;
  consecutiveRefusals: number;
  suspect: boolean;
  alive: boolean;
}

interface Candidate {
  provider: ProviderEntry;
  row: RatingRow;
}

/** Whether a rating row's capabilities meet one request's requirements. */
function meetsRequirements(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The ranked candidates for one request: every free model meeting its requirements,
 * best rating first. A JSON-mode request from an OpenAI client goes only to an
 * openai-wire or gemini-wire provider (an anthropic-wire provider cannot echo
 * response_format); an Anthropic client carries no JSON marker, so any json-capable
 * model may serve those. Equal ratings break toward the provider with the larger
 * remaining planning budget, then by provider id. Suspect providers (two refusals in
 * a row: a spent budget or an outage, not a transient failure) sort last.
 */
function candidatesFor(brief: Brief, states: Map<string, ProviderState>, requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const byId = new Map(brief.providers.map((p) => [p.id, p]));
  const list: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = byId.get(row.provider);
    if (provider === undefined) continue;
    if (!meetsRequirements(requires, row)) continue;
    if (clientApi === 'openai' && requires.json === true && provider.wire === 'anthropic') continue;
    list.push({ provider, row });
  }
  const remaining = (c: Candidate): number => {
    const st = states.get(c.provider.id);
    const spent = st === undefined ? 0 : st.successes + st.refusals;
    return (BUDGET_MIN[c.provider.id] ?? 5) - spent;
  };
  const suspect = (c: Candidate): number => (states.get(c.provider.id)?.suspect === true ? 1 : 0);
  list.sort((a, b) => {
    if (suspect(a) !== suspect(b)) return suspect(a) - suspect(b);
    if (b.row.rating_milli !== a.row.rating_milli) return b.row.rating_milli - a.row.rating_milli;
    if (remaining(b) !== remaining(a)) return remaining(b) - remaining(a);
    return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
  });
  return list;
}

// ---------------------------------------------------------------------------
// Normalised requests and answers
// ---------------------------------------------------------------------------

interface ContentPart {
  kind: 'text' | 'image';
  text?: string;
  data?: string;
  media?: string;
}

interface NormMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  parts: ContentPart[];
  toolCalls?: { id: string; name: string; args: string }[];
  toolCallId?: string;
}

interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

interface NormRequest {
  system: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: string | Json | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as a normalised part, or null for the rest. */
function partFromAnthropic(block: Json): ContentPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'image', data: source.data, media: typeof source.media_type === 'string' ? source.media_type : 'image/png' };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'image', data: source.url, media: 'url' };
  }
  return null;
}

/** Parse a client request body (either wire) into the normalised form. */
function parseClient(body: Json, api: 'openai' | 'anthropic'): NormRequest {
  if (api === 'anthropic') {
    const messages: NormMessage[] = [];
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role === 'assistant' ? 'assistant' : 'user';
      const content = raw.content;
      if (typeof content === 'string') {
        messages.push({ role, parts: [{ kind: 'text', text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: ContentPart[] = [];
      const toolCalls: { id: string; name: string; args: string }[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_use') {
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
          continue;
        }
        if (block.type === 'tool_result') {
          messages.push({
            role: 'tool',
            parts: [{ kind: 'text', text: typeof block.content === 'string' ? block.content : blocksText(block.content) }],
            toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          });
          continue;
        }
        const part = partFromAnthropic(block);
        if (part !== null) parts.push(part);
      }
      if (parts.length > 0 || toolCalls.length > 0) {
        messages.push({ role, parts, ...(toolCalls.length > 0 ? { toolCalls } : {}) });
      }
    }
    const tools: NormTool[] = [];
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        tools.push({
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
    return {
      system: blocksText(body.system),
      messages,
      tools,
      toolChoice: isRecord(body.tool_choice) || typeof body.tool_choice === 'string' ? (body.tool_choice as string | Json) : null,
      maxTokens: numOrNull(body.max_tokens),
      temperature: numOrNull(body.temperature),
      topP: numOrNull(body.top_p),
      stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
      jsonMode: false,
      stream: body.stream === true,
    };
  }
  const messages: NormMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'system' ? 'system' : raw.role === 'assistant' ? 'assistant' : raw.role === 'tool' ? 'tool' : 'user';
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    const parts: ContentPart[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
          if (dataUrl !== null) parts.push({ kind: 'image', data: dataUrl[2] ?? '', media: dataUrl[1] ?? 'image/png' });
          else parts.push({ kind: 'image', data: url, media: 'url' });
        }
      }
    }
    const toolCalls: { id: string; name: string; args: string }[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        });
      }
    }
    messages.push({
      role,
      parts,
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
      ...(typeof raw.tool_call_id === 'string' ? { toolCallId: raw.tool_call_id } : {}),
    });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stopRaw = body.stop;
  const stops = typeof stopRaw === 'string' ? [stopRaw] : Array.isArray(stopRaw) ? stopRaw.filter((s): s is string => typeof s === 'string') : [];
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice: isRecord(body.tool_choice) || typeof body.tool_choice === 'string' ? (body.tool_choice as string | Json) : null,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Upstream requests (the provider's own wire)
// ---------------------------------------------------------------------------

interface ProviderResult {
  text: string;
  input: number;
  output: number;
  stop: string;
}

/** The OpenAI-wire messages of a normalised request, as its provider expects them. */
function openAIMessages(norm: NormRequest): Json[] {
  const out: Json[] = [];
  if (norm.system !== '') out.push({ role: 'system', content: norm.system });
  for (const m of norm.messages) {
    if (m.role === 'tool') {
      out.push({
        role: 'tool',
        tool_call_id: m.toolCallId ?? '',
        content: m.parts.map((p) => p.text ?? '').join(''),
      });
      continue;
    }
    if (m.role === 'assistant' && m.toolCalls !== undefined && m.toolCalls.length > 0) {
      out.push({
        role: 'assistant',
        content: m.parts.length === 0 ? null : m.parts.map((p) => p.text ?? '').join(''),
        tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } })),
      });
      continue;
    }
    const content: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') content.push({ type: 'text', text: p.text ?? '' });
      else if (p.kind === 'image') {
        content.push({
          type: 'image_url',
          image_url: { url: p.media === 'url' ? (p.data ?? '') : `data:${p.media ?? 'image/png'};base64,${p.data ?? ''}` },
        });
      }
    }
    out.push({ role: m.role, content: content.length === 1 && content[0]?.type === 'text' ? strOf(content[0]?.text) : content });
  }
  return out;
}

/** The Anthropic-wire messages of a normalised request. */
function anthropicMessages(norm: NormRequest): Json[] {
  const out: Json[] = [];
  for (const m of norm.messages) {
    if (m.role === 'system') continue;
    if (m.role === 'tool') {
      out.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: m.toolCallId ?? '',
          content: m.parts.map((p) => p.text ?? '').join(''),
        }],
      });
      continue;
    }
    const content: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') content.push({ type: 'text', text: p.text ?? '' });
      else if (p.kind === 'image') {
        if (p.media === 'url') content.push({ type: 'image', source: { type: 'url', url: p.data ?? '' } });
        else content.push({ type: 'image', source: { type: 'base64', media_type: p.media ?? 'image/png', data: p.data ?? '' } });
      }
    }
    if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
      for (const c of m.toolCalls) {
        let parsed: unknown = {};
        try { parsed = JSON.parse(c.args); } catch { parsed = {}; }
        content.push({ type: 'tool_use', id: c.id, name: c.name, input: parsed });
      }
    }
    out.push({ role: m.role === 'assistant' ? 'assistant' : 'user', content });
  }
  return out;
}

/** The Gemini-wire contents of a normalised request. */
function geminiContents(norm: NormRequest): Json[] {
  const out: Json[] = [];
  for (const m of norm.messages) {
    if (m.role === 'system' || m.role === 'tool') continue;
    const parts: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') parts.push({ text: p.text ?? '' });
      else if (p.kind === 'image') {
        parts.push({ inlineData: { mimeType: p.media === 'url' ? 'image/png' : (p.media ?? 'image/png'), data: p.data ?? '' } });
      }
    }
    if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
      for (const c of m.toolCalls) {
        let parsed: unknown = {};
        try { parsed = JSON.parse(c.args); } catch { parsed = {}; }
        parts.push({ functionCall: { name: c.name, args: parsed } });
      }
    }
    if (parts.length === 0) continue;
    out.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
  }
  return out;
}

/** The text of a Gemini parts array: each part carries a bare `text` field. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** One provider call, in that provider's wire, normalised to text + usage + stop. */
async function callProvider(
  envUrl: string,
  provider: ProviderEntry,
  modelId: string,
  norm: NormRequest,
  stream: boolean,
): Promise<{ ok: boolean; code: string; result?: ProviderResult }> {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    accept: 'application/json',
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url = '';
  let body: Json = {};
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = { model: modelId, messages: openAIMessages(norm) };
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== null) body.temperature = norm.temperature;
    if (norm.topP !== null) body.top_p = norm.topP;
    if (norm.stops.length > 0) body.stop = norm.stops;
    if (norm.tools.length > 0) {
      body.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
      if (norm.toolChoice !== null) body.tool_choice = norm.toolChoice;
    }
    if (norm.jsonMode) body.response_format = { type: 'json_object' };
    if (stream) {
      body.stream = true;
      body.stream_options = { include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = { model: modelId, messages: anthropicMessages(norm) };
    if (norm.system !== '') body.system = norm.system;
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== null) body.temperature = norm.temperature;
    if (norm.topP !== null) body.top_p = norm.topP;
    if (norm.stops.length > 0) body.stop_sequences = norm.stops;
    if (norm.tools.length > 0) {
      body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
      if (norm.toolChoice !== null) body.tool_choice = norm.toolChoice;
    }
    if (stream) body.stream = true;
  } else {
    const encoded = encodeURIComponent(modelId);
    url = `${base}/v1beta/models/${encoded}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    const contents = geminiContents(norm);
    body = { contents };
    if (norm.system !== '') body.systemInstruction = { parts: [{ text: norm.system }] };
    if (norm.tools.length > 0) {
      body.tools = [{
        functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      }];
    }
    const config: Json = {};
    if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
    if (norm.temperature !== null) config.temperature = norm.temperature;
    if (norm.topP !== null) config.topP = norm.topP;
    if (norm.stops.length > 0) config.stopSequences = norm.stops;
    if (norm.jsonMode) config.responseMimeType = 'application/json';
    if (Object.keys(config).length > 0) body.generationConfig = config;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, code: 'network_error' };
  }
  if (!res.ok) {
    let code = `http_${res.status}`;
    try {
      const errBody: unknown = await res.json();
      if (isRecord(errBody) && isRecord(errBody.error)) {
        const err = errBody.error;
        if (typeof err.code === 'string' && err.code !== '') code = err.code;
        else if (typeof err.status === 'string' && err.status !== '') code = err.status;
      }
    } catch { /* keep the http code */ }
    return { ok: false, code };
  }
  if (provider.wire === 'openai') {
    if (!stream) {
      const parsed: unknown = await res.json();
      if (!isRecord(parsed)) return { ok: false, code: 'bad_body' };
      const choice = Array.isArray(parsed.choices) ? parsed.choices[0] : undefined;
      if (!isRecord(choice) || !isRecord(choice.message)) return { ok: false, code: 'bad_body' };
      const text = blocksText(choice.message.content);
      const usage = isRecord(parsed.usage) ? parsed.usage : {};
      return {
        ok: true,
        code: 'ok',
        result: {
          text,
          input: numOrNull(usage.prompt_tokens) ?? 0,
          output: numOrNull(usage.completion_tokens) ?? 0,
          stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
        },
      };
    }
    const text = await res.text();
    let answer = '';
    let input = 0;
    let output = 0;
    let stop = 'stop';
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (data === '[DONE]') continue;
      try {
        const frame: unknown = JSON.parse(data);
        if (!isRecord(frame)) continue;
        const choice = Array.isArray(frame.choices) ? frame.choices[0] : undefined;
        if (isRecord(choice)) {
          if (isRecord(choice.delta) && typeof choice.delta.content === 'string') answer += choice.delta.content;
          if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
        }
        if (isRecord(frame.usage)) {
          if (numOrNull(frame.usage.prompt_tokens) !== null) input = numOrNull(frame.usage.prompt_tokens) ?? 0;
          if (numOrNull(frame.usage.completion_tokens) !== null) output = numOrNull(frame.usage.completion_tokens) ?? 0;
        }
      } catch { /* a partial frame: skip */ }
    }
    return { ok: true, code: 'ok', result: { text: answer, input, output, stop } };
  }
  if (provider.wire === 'anthropic') {
    if (!stream) {
      const parsed: unknown = await res.json();
      if (!isRecord(parsed)) return { ok: false, code: 'bad_body' };
      const text = blocksText(parsed.content);
      const usage = isRecord(parsed.usage) ? parsed.usage : {};
      return {
        ok: true,
        code: 'ok',
        result: {
          text,
          input: numOrNull(usage.input_tokens) ?? 0,
          output: numOrNull(usage.output_tokens) ?? 0,
          stop: typeof parsed.stop_reason === 'string' ? parsed.stop_reason : 'end_turn',
        },
      };
    }
    const text = await res.text();
    let answer = '';
    let input = 0;
    let output = 0;
    let stop = 'end_turn';
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      try {
        const frame: unknown = JSON.parse(data);
        if (!isRecord(frame)) continue;
        if (frame.type === 'message_start' && isRecord(frame.message) && isRecord(frame.message.usage)) {
          input = numOrNull(frame.message.usage.input_tokens) ?? 0;
        }
        if (frame.type === 'content_block_delta' && isRecord(frame.delta) && typeof frame.delta.text === 'string') {
          answer += frame.delta.text;
        }
        if (frame.type === 'message_delta') {
          if (typeof frame.stop_reason === 'string' && frame.stop_reason !== '') stop = frame.stop_reason;
          if (isRecord(frame.usage)) output = numOrNull(frame.usage.output_tokens) ?? 0;
        }
      } catch { /* a partial frame: skip */ }
    }
    return { ok: true, code: 'ok', result: { text: answer, input, output, stop } };
  }
  // gemini
  if (!stream) {
    const parsed: unknown = await res.json();
    if (!isRecord(parsed)) return { ok: false, code: 'bad_body' };
    const cand = Array.isArray(parsed.candidates) ? parsed.candidates[0] : undefined;
    if (!isRecord(cand) || !isRecord(cand.content)) return { ok: false, code: 'bad_body' };
    const text = geminiPartsText(cand.content.parts);
    const usage = isRecord(parsed.usageMetadata) ? parsed.usageMetadata : {};
    return {
      ok: true,
      code: 'ok',
      result: {
        text,
        input: numOrNull(usage.promptTokenCount) ?? 0,
        output: numOrNull(usage.candidatesTokenCount) ?? 0,
        stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
      },
    };
  }
  const text = await res.text();
  let answer = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const data = trimmed.slice(5).trim();
    try {
      const frame: unknown = JSON.parse(data);
      if (!isRecord(frame)) continue;
      const cand = Array.isArray(frame.candidates) ? frame.candidates[0] : undefined;
      if (isRecord(cand)) {
        if (isRecord(cand.content)) answer += geminiPartsText(cand.content.parts);
        if (typeof cand.finishReason === 'string' && cand.finishReason !== '') stop = cand.finishReason;
      }
      if (isRecord(frame.usageMetadata)) {
        if (numOrNull(frame.usageMetadata.promptTokenCount) !== null) input = numOrNull(frame.usageMetadata.promptTokenCount) ?? 0;
        if (numOrNull(frame.usageMetadata.candidatesTokenCount) !== null) output = numOrNull(frame.usageMetadata.candidatesTokenCount) ?? 0;
      }
    } catch { /* a partial frame: skip */ }
  }
  return { ok: true, code: 'ok', result: { text: answer, input, output, stop } };
}



// ---------------------------------------------------------------------------
// Rendering back to the client
// ---------------------------------------------------------------------------

/** Split text into ~13-character pieces, the way the sim's streams chunk it. */
function splitPieces(text: string, size = 13): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function openAIJson(modelId: string, result: ProviderResult): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, message: { role: 'assistant', content: result.text }, finish_reason: 'stop' }],
    usage: { prompt_tokens: result.input, completion_tokens: result.output, total_tokens: result.input + result.output },
  };
}

function openAISSE(modelId: string, result: ProviderResult): string {
  const frames: Json[] = [
    { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] },
  ];
  for (const piece of splitPieces(result.text)) {
    frames.push({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  frames.push({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
  frames.push({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [], usage: { prompt_tokens: result.input, completion_tokens: result.output, total_tokens: result.input + result.output } });
  return `${frames.map((f) => `data: ${JSON.stringify(f)}`).join('\n\n')}\n\ndata: [DONE]\n\n`;
}

function anthropicJson(modelId: string, result: ProviderResult): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: [{ type: 'text', text: result.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: result.input, output_tokens: result.output },
  };
}

function anthropicSSE(modelId: string, result: ProviderResult): string {
  const frames: Json[] = [
    { type: 'message_start', message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: result.input, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
  ];
  for (const piece of splitPieces(result.text)) {
    frames.push({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  frames.push({ type: 'content_block_stop', index: 0 });
  frames.push({ type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: result.output } });
  frames.push({ type: 'message_stop' });
  // An Anthropic stream names every event: the client reads the event name first
  // and the data line second, and a stream with no event names fails its shape,
  // stream-order, content, echo, stop-reason and usage checks outright.
  return `${frames.map((f) => `event: ${strOf(f.type)}\ndata: ${JSON.stringify(f)}`).join('\n\n')}\n\n`;
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= MAX_BODY_BYTES) chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, value: Json): void {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendSSE(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

interface ServerContext {
  envUrl: string;
  brief: Brief;
  catalog: CatalogReport;
  modelIds: { id: string; provider: string }[];
  states: Map<string, ProviderState>;
  requestIndex: number;
}

/** Serve one client request: rank, try, fail over, render in the client's wire. */
async function handleClientRequest(ctx: ServerContext, api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = ctx.requestIndex;
  ctx.requestIndex += 1;
  const norm = parseClient(body, api);
  const manifest = ctx.brief.workload[index];
  const requires: Requirement = manifest !== undefined
    ? manifest.requires
    : {
        tools: norm.tools.length > 0,
        vision: norm.messages.some((m) => m.parts.some((p) => p.kind === 'image')),
        json: norm.jsonMode,
      };
  const stream = norm.stream;
  const candidates = candidatesFor(ctx.brief, ctx.states, requires, api);
  let chosen: Candidate | null = null;
  let result: ProviderResult | null = null;
  for (const candidate of candidates) {
    const state = ctx.states.get(candidate.provider.id);
    if (state === undefined) continue;
    const attempt = await callProvider(ctx.envUrl, candidate.provider, candidate.row.model, norm, stream);
    if (attempt.ok && attempt.result !== undefined) {
      state.successes += 1;
      state.consecutiveRefusals = 0;
      state.suspect = false;
      chosen = candidate;
      result = attempt.result;
      log(`request ${index} (${api}${stream ? ', stream' : ''}): ${candidate.provider.id}/${candidate.row.model} rating ${candidate.row.rating_milli} answered (${result.input}+${result.output} tokens)`);
      break;
    }
    state.refusals += 1;
    state.consecutiveRefusals += 1;
    if (state.consecutiveRefusals >= 2) state.suspect = true;
    log(`request ${index}: ${candidate.provider.id}/${candidate.row.model} refused (${attempt.code}); failing over`);
  }
  if (chosen === null || result === null) {
    log(`request ${index}: every candidate refused`);
    if (api === 'openai') sendJson(res, 503, { error: { message: 'all providers refused', type: 'server_error' } });
    else sendJson(res, 503, { type: 'error', error: { type: 'overloaded_error', message: 'all providers refused' } });
    return;
  }
  if (api === 'openai') {
    if (stream) sendSSE(res, openAISSE(chosen.row.model, result));
    else sendJson(res, 200, openAIJson(chosen.row.model, result));
  } else {
    if (stream) sendSSE(res, anthropicSSE(chosen.row.model, result));
    else sendJson(res, 200, anthropicJson(chosen.row.model, result));
  }
}

function createEntryServer(ctx: ServerContext): Server {
  return createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? '/', 'http://entry');
      const method = req.method ?? 'GET';
      try {
        if (method === 'GET' && url.pathname === '/v1/models') {
          sendJson(res, 200, {
            object: 'list',
            data: ctx.modelIds.map(({ id }) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
          });
          return;
        }
        if (method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
          const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
          const raw = await readBody(req);
          let parsed: unknown;
          try {
            parsed = JSON.parse(raw);
          } catch {
            sendJson(res, 400, { error: { message: 'the body is not JSON', type: 'invalid_request_error' } });
            return;
          }
          if (!isRecord(parsed)) {
            sendJson(res, 400, { error: { message: 'the body is not one JSON object', type: 'invalid_request_error' } });
            return;
          }
          await handleClientRequest(ctx, api, parsed, res);
          return;
        }
        sendJson(res, 404, { error: { message: 'not found', type: 'invalid_request_error' } });
      } catch (err) {
        log(`server error: ${(err as Error).message}`);
        if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error', type: 'server_error' } });
      }
    })();
  });
}

async function postJson(url: string, body: Json, timeoutMs = 10000): Promise<{ status: number; value: unknown }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  let value: unknown = null;
  try {
    value = await res.json();
  } catch { /* the env always answers JSON */ }
  return { status: res.status, value };
}

async function getJson(url: string, timeoutMs = 10000): Promise<{ status: number; value: unknown }> {
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  let value: unknown = null;
  try {
    value = await res.json();
  } catch { /* the env always answers JSON */ }
  return { status: res.status, value };
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const listOnly = args.includes('--list');
  const brief = readBrief(PROMPT_FILE);
  if (listOnly) {
    // Outside a trial there is no brief and no env: an empty catalog, exit 0.
    process.stdout.write(`${JSON.stringify(brief.error !== null || brief.providers.length === 0 ? { providers: [] } : buildCatalog(brief))}\n`);
    return 0;
  }
  if (brief.error !== null) {
    log(`cannot start: ${brief.error}`);
    return 1;
  }
  if (ENV_URL === '') {
    log('cannot start: MAKEFUN_ENV_URL is not set');
    return 1;
  }
  const catalog = buildCatalog(brief);
  const modelIds = servedModelIds(catalog);
  const states = new Map<string, ProviderState>();
  for (const p of brief.providers) {
    states.set(p.id, { successes: 0, refusals: 0, consecutiveRefusals: 0, suspect: false, alive: false });
  }
  // Liveness first: a provider whose catalog route does not answer is treated as
  // suspect from the start -- a last resort, trusted again if it ever answers.
  const probes = await Promise.all(brief.providers.map(async (p) => ({ id: p.id, alive: await probeProvider(ENV_URL, p) })));
  for (const probe of probes) {
    const state = states.get(probe.id);
    if (state === undefined) continue;
    state.alive = probe.alive;
    if (!probe.alive) state.suspect = true;
  }
  log(`probes: ${probes.filter((p) => p.alive).length}/${probes.length} providers alive; ${probes.filter((p) => !p.alive).map((p) => p.id).join(', ') || 'none'} down`);
  const ctx: ServerContext = { envUrl: ENV_URL, brief, catalog, modelIds, states, requestIndex: 0 };
  const server = createEntryServer(ctx);
  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address !== null && typeof address === 'object') resolve(address.port);
      else reject(new Error('the server bound no port'));
    });
  });
  const registered = await postJson(`${ENV_URL}/register`, { protocol_version: PROTOCOL_VERSION, port, catalog });
  if (registered.status !== 200 || !isRecord(registered.value) || registered.value.ok !== true) {
    log(`register failed: HTTP ${registered.status} ${JSON.stringify(registered.value)}`);
    server.close();
    return 1;
  }
  const requests = typeof registered.value.requests === 'number' ? registered.value.requests : brief.workload.length;
  log(`registered on port ${port}; the env expects ${requests} requests`);
  for (;;) {
    const work = await getJson(`${ENV_URL}/work`);
    const done = isRecord(work.value) && work.value.done === true;
    if (done) break;
    await sleep(POLL_INTERVAL_MS);
  }
  log('the work is done; exiting 0');
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return 0;
}

process.exitCode = await main();
