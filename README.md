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
# CHATGPT_INCLUDE_SOURCES=0        # append Sources block for llmcord-go Show Sources button
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
- Failover: if a request fails on one account (429/403/5xx/transport error), EVERY other available account is tried exactly once before an error is returned. Attempts served by a different account rebuild the full conversation from scratch (see "Multi-turn design"), so nothing is lost — text, images, and binary files alike are replayed onto the account that actually serves the turn.
- Image-limit refusals: when ChatGPT answers with "You've hit the ... plan limit for image generations requests ...", that account is put on cooldown like a 429 and the request is retried on the next available account — before any of the refusal text reaches your client. Only if every account is image-limited does the request fail (429), carrying the last refusal message.
- Image generation and editing login refusals: if ChatGPT answers with "Image generation and image editing require you to be logged in...", the proxy treats it as an image refusal on that account, cooling it down and seamlessly failing over to another account.
- Session cookies authenticate most backend-api calls even when the embedded access-token JWT has expired — but file uploads and image generations are **Bearer-authenticated** (backend-api / conversation image generation requires an active authenticated session). Tokens refresh opportunistically via `/api/auth/session`; if that endpoint keeps returning a token that is already expired, the session cookie itself is stale. Such accounts log `needs re-login` and skip uploads immediately instead of failing mid-upload — replace their block in `accounts.txt` with a fresh cookie-jar + session-JSON export to restore image/file requests.
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

**Account failover & context preservation.** Each response records a snapshot of
the complete client-visible context (every turn's text plus its images/files,
capped per response and store-wide via `SNAPSHOT_FILE_CAP_MB` /
`SNAPSHOT_STORE_CAP_MB`). If the owning account fails mid-request, the next
available account serves the turn in a brand-new conversation: prior turns are
rendered into the prompt (`[user]` / `[assistant]` blocks, system included)
and every attachment is re-uploaded to that account, because uploads are
account-scoped. Only after all available accounts have been tried does the
request fail, carrying the last underlying error. Output already streamed to
the client is never duplicated: once a delta has been emitted, no further
account switching happens.
Image-generation quota refusals ("You've hit the Free plan limit for image
generations requests...") are detected inside the stream and withheld from the

client while the reply could still be such a refusal (normally a few
characters of added latency, at most ~220 while divergence is unproven); a
matching reply rotates to the next account instead of being delivered. Replies that merely begin similarly ("You've hit the nail on
the head!") stream normally.
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
### Show Sources (llmcord-go)

To feed search citations into `llmcord-go`'s **Show Sources** button:
- Enable globally by setting `CHATGPT_INCLUDE_SOURCES=1` (or `INCLUDE_SOURCES=1`) in `.env`.
- Or enable per request with `"include_sources": true` in the JSON request body (e.g. in `llmcord-go`, set `extra_body: {include_sources: true}`).

When enabled and ChatGPT uses web search, an appendix formatted as:
```markdown
Sources
1. [Title](url) (domain) via `query`

Search Queries
1. `query`
```
is appended at the end of the turn. `llmcord-go` automatically hides this appendix from the visible message while rendering and parses it to populate the **Show Sources** button and paginated sources view.

## Not supported

Tool/function calling, audio input, assistant tool-call replay, server-side `file_id` references. Requests containing them get a clear 400.

## Limits

Free ChatGPT accounts have small message/image quotas; the pool's cooldowns absorb 429s. The Plus account re-enters rotation fastest. Don't hammer the free accounts with image generation.
