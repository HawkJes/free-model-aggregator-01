# free-model-aggregator-01 — rating-first aggregator with failover

One server that serves the OpenAI Chat Completions API and the Anthropic Messages
API over ten simulated free-tier model providers, in JSON and in server-sent events,
and registers the catalog it read from them. Entry for the make.fun competition
`free-model-aggregator-01` (Node 22, standard library only, single file `main.ts`).

## Run it

```sh
node main.ts            # read the brief, probe the providers, register, serve
node main.ts --list     # print the catalog document and exit 0
```

A trial sets `MAKEFUN_PROMPT_FILE` (the brief: provider directory, quality table,
workload manifest) and `MAKEFUN_ENV_URL` (the env serving the providers). The server
listens on an ephemeral port, registers it with `POST $MAKEFUN_ENV_URL/register`,
serves the twenty requests, and exits 0 when `GET $MAKEFUN_ENV_URL/work` says the
workload is done. Outside a trial, `--list` prints `{"providers":[]}`.

## How it routes

- **Rating first.** Every free model is ranked by the brief's quality table
  (`rating_milli`), best first. Vision requests go to the best vision-capable model;
  requests needing a long context go to the best model whose context covers it.
- **JSON mode stays on a wire that can carry it.** A JSON-mode request (OpenAI
  `response_format`) only goes to an openai-wire or gemini-wire provider, because an
  anthropic-wire provider would echo `json_mode: false`.
- **Failover down the ranked list.** A provider that answers 402, 429, 5xx, or never
  answers costs the request nothing: the next candidate serves it. A provider that
  refuses twice in a row is demoted to last resort for the rest of the trial, and a
  provider whose catalog probe failed starts there.
- **Full wire normalisation.** System text, user text, images by their bytes, tools
  in order, `max_tokens`, stop sequences, and JSON mode are carried between the client
  wire (OpenAI / Anthropic) and the provider wire (openai / anthropic / gemini)
  untouched. The provider's text comes back whole, its usage counts as reported, and
  its stop reason is mapped (`stop` ↔ `end_turn` ↔ `STOP`, `length` ↔ `max_tokens`,
  `tool_calls` ↔ `tool_use`, `content_filter` ↔ `refusal`).
- **Streaming is translated, not just forwarded.** Same-wire streams pass through
  verbatim; cross-wire streams are re-emitted in the client's event order, with the
  provider's usage in the final chunk (OpenAI) or `message_delta` (Anthropic).
  Anthropic SSE frames always carry their `event:` names.

## Layout

- `main.ts` — the whole server: brief parsing, catalog probes, routing, the three
  provider wires, both client wires, registration, and the work poll.
- `LICENSE` — MIT.
- `package.json` — `{"type": "module"}`.
- `.makefun/owner.json` — the owner's wallet claim.

## Typecheck

```sh
~/workspace/makefun/node_modules/.bin/tsc --noEmit --target es2022 --module nodenext --moduleResolution nodenext --strict main.ts
```
