# Wald-Q4B decision API

Wald-Q4B's server, `wald-serve`, exposes one decision endpoint, `POST /v1/systemone`. Its request and answer shapes follow TypeSafe's `/v1/systemone` format, so clients written for Jev can point at a self-hosted Wald server. Wald is independent and is not affiliated with TypeSafe AI.

Start the server with `./run.sh "$PWD"` in the downloaded model directory ([model card](https://huggingface.co/org2ai/Wald-4B), [RUNBOOK.md](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md)). It listens on port 8000 and does not check API keys.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/systemone` (also `POST /`) | Answer one or more typed questions about a state |
| `GET /health` | Effective policy: effort, gate, thought budget, prompt format, context limit |
| `GET /v1/models` | The served model name |

## Request

| Field | Type | Meaning |
|---|---|---|
| `state` | string, object, array or null | What the decision is about: a message, a conversation, a document, an agent trace. Objects and arrays are flattened to text with their field names kept. |
| `questions` | object, at least one entry | Question id → question. Every question is answered about the same `state`. |
| `effort` | string, optional | `none`, `low`, `medium`, `high` or `high-k2` … `high-k8`. Overrides the server default for this request. |
| `model` | string, optional | Accepted for client compatibility; the server answers with the model it serves. |

Each question has a `type`, optional `instructions` (any JSON, usually a sentence) and `criteria`:

| `type` | `criteria` | Answer |
|---|---|---|
| `choice` | Object of 1–255 options: key → description (description may be null) | `choice` (the most probable key), `probabilities` (key → probability), `confidence` |
| `noul` | Optional object with `true` and/or `false` descriptions | `noul`: the probability of yes |
| `score` | Array of 1–255 ordered levels | `score` (expected level index), `probabilities` (level index → probability), `confidence` |

Every answer also carries `mode`: `A` for a one-pass read, `B` for a read after a thought, `K` for grouped (knockout) reading of wide option sets.

Prompts longer than the context limit (131,072 tokens by default) are rejected with HTTP 422, never truncated. A malformed request or unknown effort returns HTTP 400.

## Example: route a tool call

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "user": "What will the weather be in Lisbon tomorrow afternoon?",
      "tools_available": ["web_search", "weather_api", "calendar"]
    },
    "effort": "none",
    "questions": {
      "tool": {
        "type": "choice",
        "instructions": "Which tool should the agent call next?",
        "criteria": {
          "web_search": "General web search",
          "weather_api": "Forecast for a city and time",
          "calendar": "Read or create calendar events",
          "none": "Answer directly without a tool"
        }
      }
    }
  }'
```

Response shape (the numbers here are illustrative, not a measured output):

```json
{
  "model": "wald-4b",
  "answers": {
    "tool": {
      "type": "choice",
      "choice": "weather_api",
      "probabilities": {"web_search": 0.04, "weather_api": 0.94, "calendar": 0.0, "none": 0.02},
      "confidence": 0.92,
      "mode": "A"
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 0},
  "latency_s": 0.03
}
```

For `choice`, `confidence` rescales the top probability so that 0 means uniform and 1 means certain: `(p_top − 1/K) / (1 − 1/K)` for K options.

## Example: decide whether to ask the user

Ask several questions about one state in a single request. Each question is read separately.

```json
{
  "state": "User: book me a table for Friday",
  "effort": "medium",
  "questions": {
    "specific_enough": {
      "type": "noul",
      "instructions": "Is the request specific enough to act on without asking a follow-up question?",
      "criteria": {"true": "Enough detail to act", "false": "Needs clarification first"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

Response shape (illustrative numbers):

```json
{
  "model": "wald-4b",
  "answers": {
    "specific_enough": {"type": "noul", "noul": 0.08, "mode": "B"},
    "urgency": {"type": "score", "score": 1.1, "probabilities": {"0": 0.15, "1": 0.6, "2": 0.25}, "confidence": 0.6, "mode": "A"}
  },
  "usage": {"input_tokens": 310, "output_tokens": 96},
  "latency_s": 0.6
}
```

A typical policy: act when `noul` is above a high threshold, ask the user when it is below a low one, and escalate the cases in between. Set both thresholds on validation data from your own task.

## Choosing an effort

| Effort | Use it for |
|---|---|
| `none` | Lowest latency. One pass, no generated tokens. |
| `low` / `medium` | Think only when the top initial probability is below 0.5 / 0.7. |
| `high` | Think on every eligible question. The published Decision Index score uses this setting. |
| `high-k2` … `high-k8` | Several thoughts, averaged. Slowest. |

Thinking applies to questions with 2–26 options when the context has room; otherwise the one-pass answer is returned.

## Python client

```python
import requests

r = requests.post("http://localhost:8000/v1/systemone", json={
    "state": "Refund request for order 1182; the parcel arrived damaged.",
    "effort": "none",
    "questions": {"route": {"type": "choice", "criteria": {
        "refunds": "Refunds and returns", "shipping": "Delivery problems", "other": "Anything else"}}},
}, timeout=30)
answer = r.json()["answers"]["route"]
print(answer["choice"], answer["probabilities"])
```

Source: [`server/src/wald_serve/wire.py`](https://huggingface.co/org2ai/Wald-4B/blob/main/server/src/wald_serve/wire.py) (request shapes) and [`server.py`](https://huggingface.co/org2ai/Wald-4B/blob/main/server/src/wald_serve/server.py) (endpoints).
