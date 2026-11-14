# free-model-aggregator-01

A single server (Node 22, standard library only) that serves the OpenAI Chat
Completions and Anthropic Messages APIs over ten simulated free-tier providers,
routing each request to the best free model that meets its requirements and
failing over on refusal.

## How it works

- The catalog is built from the brief's quality table, which is the sealed truth
  for which models are free and what they can do. `node main.ts --list` prints it.
- Each provider's catalog route is probed for liveness at startup. A provider
  that does not answer starts as a last resort; one that refuses twice in a row
  is deprioritised for the rest of the trial.
- Routing is rating-first: vision requests go to gemini-3.8-flash (1458), the
  rest to z-ai/glm-5.3-flash:free (1475) on openrouter, then glm-4.7-flash
  (1470), deepseek-v3.2 (1465), and down the table. JSON-mode requests from an
  OpenAI client go only to an openai-wire or gemini-wire provider.
- Every request is normalised, sent in the provider's own wire with its system
  text, user text, images, tools, stops, max_tokens and JSON mode untouched, and
  the provider's answer is rendered back in the wire the client used, with the
  provider's text whole and its usage counts as reported.

## Run

```sh
node main.ts          # serve, register, exit 0 when the work is done
node main.ts --list   # print the catalog and exit 0
```
