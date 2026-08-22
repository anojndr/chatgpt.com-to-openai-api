# chatgpt-to-openai-api

Exposes ChatGPT web accounts as a local **OpenAI-compatible API** (Chat Completions + Responses) on port **4035**, using FastAPI. No browser is driven; all traffic is plain HTTPS to `chatgpt.com`'s internal backend with TLS impersonation (`curl_cffi`) and the sentinel proof-of-work handshake.

## Quick start

```bash
./run.sh                 # creates .venv if missing, serves http://0.0.0.0:4035
```

Configuration lives in `.env` (never commit real keys):

```
PIXELVAULT_API_KEY=pv_live_xxx     # required for generated-image hosting
PORT=4035
# optional:
# ACCOUNTS_FILE=accounts.txt
# API_KEY=...                      # require this bearer key on /v1/*
# DEFAULT_MODEL=auto
# COOLDOWN_FREE_SECONDS=900
# COOLDOWN_PLUS_SECONDS=120
```

## Accounts

`accounts.txt` holds any number of account blocks:

```
account 1:
<<<
netscape cookies:
<Netscape cookie jar export for chatgpt.com>

from https://chatgpt.com/api/auth/session:
<the JSON body of that endpoint>
>>>
account 2:
...
```

- Add or remove blocks while the server runs — the pool hot-reloads within ~2 s (dedup by user id).
- Load balancing: fair least-in-flight / least-recently-used rotation across ALL accounts; 429s put an account on cooldown (free 15 min, plus 2 min).
- Session cookies authenticate backend-api even when the embedded access-token JWT has expired; tokens refresh opportunistically via `/api/auth/session`.
- For tests you can pin an account with header `x-chatgpt-account: <email>`.

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | live ChatGPT slugs (`auto`, `gpt-5-6`, `gpt-5-5`, ...) |
| `POST /v1/chat/completions` | stream + non-stream, `stream_options.include_usage`, system/developer messages, text/image/file parts |
| `POST /v1/responses` | stream + non-stream, `instructions`, `previous_response_id` stateful continuation |
| `GET /v1/accounts` | pool snapshot |
| `GET /healthz` | liveness + available accounts |

### Multi-turn design

Requests that resend full history are matched against stored hash-chain prefixes; only the NEW trailing message is forwarded to ChatGPT and the real server-side conversation continues (`conversation_id` + `parent_message_id`). Logs show e.g. `turn plan: history=3 matched=1 continue conv=...`. `previous_response_id` maps to the same registry.

### Files

- Text-like inputs (`.py .json .txt .md .csv ...`) are inlined as fenced code blocks — fastest path.
- Images (`png jpeg webp gif`) upload to ChatGPT as multimodal assets (data URLs or http(s)).
- Binary docs (`pdf`, office, etc.) attach via ChatGPT's file pipeline.

### Image output

Generated images are detected in the stream (`sediment://` pointers), downloaded, uploaded to PixelVault, and appended as markdown:

```
![generated image](https://img.pixelvault.dev/proj_*/img_*.png)
```

Try: `{"model":"auto","messages":[{"role":"user","content":"generate an image of a cat"}]}`

## Not supported

Tool/function calling, audio input, assistant tool-call replay, server-side `file_id` references. Requests containing them get a clear 400.

## Limits

Free ChatGPT accounts have small message/image quotas; the pool's cooldowns absorb 429s. The Plus account re-enters rotation fastest. Don't hammer the free accounts with image generation.
