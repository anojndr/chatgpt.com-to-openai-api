# Repository Guidelines

## Project Overview

- Local OpenAI-compatible API over ChatGPT web accounts: `POST /v1/chat/completions` + `POST /v1/responses` via FastAPI on `127.0.0.1:4035`.
- No browser: plain HTTPS to `chatgpt.com/backend-api` with `curl_cffi` TLS impersonation (`chrome`) + Sentinel PoW handshake (`app/pow_solver.py`).
- Account pool from `accounts.txt` (Netscape cookie-jar + session JSON) with hot-reload + keepalive; conversation continuity in SQLite `data/conversations.db`.

## Architecture & Data Flow

- Proxy pipeline: `app/main.py` → `app/adapters.py:parse_chat_request` / `parse_responses_request` → `ParsedRequest` → `app/engine.py:run_turn` / `collect` → OpenAI SSE/JSON encode.
- Turn planning (`engine._plan`): rolling SHA-256 hash chain (`store.item_hash`) finds longest recorded prefix → resume live `conversation_id` with trailing turns only; else fresh conversation with `_replay_prompt`. `previous_response_id` replays `TurnSnapshot`.
- Attempt loop (`engine.run_turn`): prefer stored-conversation owner / `x-chatgpt-account` header, then each untried account once. `_attempt_deltas` → `_pump_stream_events` folds backend SSE into `delta` + `done(TurnResult)`; `_record_done` persists prefix + snapshot; `_resolve_images` uploads via `freeimage`, charts via `charts.render_chart_png`.
- Backend driver (`chatgpt.AccountSession` on `curl_cffi.AsyncSession`): `refresh_access_token` → proof config (`GET /?oai-dm=1`) → `POST /sentinel/chat-requirements` (PoW cached as `Requirements`) → `POST /conversation` SSE → file flow `POST /files` → `PUT` blob → `POST /files/{id}/uploaded`.
- Model resolution (`adapters.map_model`, `LIVE_SLUG_ALIASES`, `config.DEFAULT_MODEL`): exact/live slug wins, else alias (e.g. `gpt-4o` → `auto`), else default. `public_models` feeds `GET /v1/models`.
- Post-processing in `engine`: citations (PUA `cite`, JSX `<Cite>`, `grouped_webpages`) → markdown links (`render_citations`, `_clean_url` strips `utm_*`); `include_sources` appendix; image-quota refusals (`IMAGE_LIMIT_RE` + `_classify_accumulated`) raise `_ImageLimitError` for failover.

## Key Directories

- `app/` — package root, no `src/` dir (10 modules, `__init__.py` trivial):
  - `main.py` — HTTP only: auth gate, body parse, SSE framing, OpenAI envelopes.
  - `engine.py` (~2270 lines) — all turn semantics: planning, streaming, citations, charts, quota, failover.
  - `adapters.py` — pure OpenAI→internal translation; dataclasses `ParsedRequest`/`HistoryItem`/`ImageInput`/`FileInput`; no I/O except guarded remote fetch.
  - `chatgpt.py` — stateful backend client (`AccountSession`, `Requirements`, `ChatGPTError`).
  - `accounts.py` — `AccountPool` + global `POOL`; `store.py` — `ConversationStore` (WAL SQLite) + global `STORE`.
  - `config.py`, `charts.py`, `freeimage.py`, `pow_solver.py` — leaf helpers.
- `tests/` — flat, no subdirs/conftest: 5 `test_*.py` modules (~1.9k lines).
- Root — flat: `run.sh`, `restart.sh`, `README.md`, `.env*`, `accounts.txt*`, `data/`, `server.log`. No `scripts/`/`docs/`/`examples/` dirs.

## Code Exploration

Always use codebase-memory-mcp.

- Graph-first: `search_graph` for symbols, `trace_path` for callers/callees, `get_code_snippet` for source, `check_index_coverage` for cited paths — filesystem `grep` only for literal/non-code text.

## Development Commands

```bash
./run.sh                        # venv bootstrap + exec .venv/bin/python -m app.main (foreground, :4035)
./restart.sh                    # pkill + setsid nohup + poll GET /healthz (30x0.5s)
tail -f server.log              # follow detached logs
uv run pytest                   # full suite (testpaths=tests, pythonpath=".")
uv run pytest tests/test_store.py
uv run pytest tests/test_keepalive.py::TestRefreshStrikes::test_strikes_mark_dead_then_success_revives
uvx ruff check .                # lint (select ALL, preview)
uvx ruff format --check .       # format check
uvx ty check                    # typecheck (all rules = error)
HOST=0.0.0.0 ./run.sh           # LAN expose (default HOST=127.0.0.1)
```

- No build step: run directly as `.venv/bin/python -m app.main`; base URL `http://127.0.0.1:$PORT/v1`.
- Bootstrap fallback without `uv`: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Code Conventions & Common Patterns

- Formatting/lint: Ruff `select=["ALL"]`, `preview=true`, Google pydocstyle; ignore only `incorrect-blank-line-before-class`, `multi-line-summary-second-line`, `missing-trailing-comma`. `ty` maximally strict (`all="error"`, `error-on-warning=true`). Google-style docstrings on public functions; `from __future__ import annotations`.
- Naming: `SCREAMING_SNAKE` constants with units (`*_SECONDS`, `*_TTL`, `*_CAP_MB`, e.g. `COOLDOWN_FREE_SECONDS`, `SNAPSHOT_STORE_CAP_MB`); files `test_<area>.py`, classes `Test<Area>`, methods `test_<behavior>`; private helpers `_plan`, `_attempt_deltas`, `_ingest_event`, `_opt_str`/`_as_object_list`.
- Error handling: `EngineError(status,message,error_type)` is only client-facing (mapped in `main.engine_error_handler`/`oai_error`); `ChatGPTError`, `RequestException`/`OSError`, `FreeimageError`, `_ImageLimitError` are attempt-level → `_failure_for` for cooldown + failover. `invalid_request_error` short-circuits (no retry); `produced=True` salvages streamed bytes instead of failover. Defensive JSON: `_require_str|dict`, `_optional_str|bool`, `TypeGuard _is_dict`, broad `except (ValueError,KeyError,TypeError,AttributeError)` on external payloads; skip malformed cookies/snapshots with log, never raise.
- Async: async-first I/O (`asyncio`, `AsyncIterator` deltas, `curl_cffi.AsyncSession`); tests drive async via `asyncio.run(...)` (no `pytest-asyncio`). SQLite sync behind `threading.RLock` + thread-local conns, WAL + `busy_timeout=30000`, explicit `BEGIN IMMEDIATE`.
- Dependency injection: globals `accounts.POOL` / `store.STORE` created at import, but core logic takes `pool` explicitly — e.g. `run_turn(parsed, pool, *, previous_response_id, preferred_email)`, `ConversationStore(db_path=None)` defaults to `config.DB_PATH`. New logic must keep this seam for `FakeAccount` tests.
- State management: LB = least-in-flight then least-recently-used; per-plan cooldowns (`COOLDOWN_FREE_SECONDS=900`, `COOLDOWN_PLUS_SECONDS=120`); `dead` + `cooldown_until` + `KEEPALIVE_MAX_STRIKES=3` with `revive_after`; always `pool.release(acct)` in `finally`. Attachments re-uploaded per attempt (`_upload_inputs`) so failover never loses content; text files inlined as fenced blocks (`fence`, `CODE_EXTS`).
- Examples: add model alias in `adapters.LIVE_SLUG_ALIASES`; add route in `app/main.py` + translate in `adapters.py` + orchestrate in `engine.py`; pin account in tests/requests via `x-chatgpt-account: <email>` header.

## Lint & Typecheck Policy

Always use https://docs.astral.sh/ruff/ with everything enabled and https://docs.astral.sh/ty/ with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.

- Ruff: `select = ["ALL"]` + `preview = true` in `pyproject.toml`; `uvx ruff check .` and `uvx ruff format --check .` must pass. Only documented tool-conflict excludes allowed (`missing-trailing-comma` redundant with formatter, `S101` scoped off in `tests/**` for `PT009`, incompatible `D203/D211` + `D212/D213` pairs where Ruff auto-disables one side) — never add `noqa`.
- ty: `[tool.ty.rules] all = "error"` + strict flags + `error-on-warning = true`; `uvx ty check` must pass. Never add `ty: ignore`, `type: ignore`, or `@no_type_check` — fix the types.

## Important Files

- Entry: `app/main.py:61 lifespan` (starts `POOL.start_watcher`), `:362 chat_completions`, `:643 responses_api`, `:147 list_models`, `:693 healthz`, `:706 accounts_snapshot`, `:70 auth_ok`, `:110 engine_error_handler`.
- Core: `app/engine.py:2140 run_turn`, `:2241 collect`, `:52 TurnResult`, `:68 EngineError`; `app/adapters.py:23 LIVE_SLUG_ALIASES`, `:41 map_model`, `:626 parse_chat_request`, `:774 parse_responses_request`; `app/chatgpt.py:88 parse_accounts_text`, `:270 AccountSession`; `app/accounts.py:105 AccountPool`, `:545 POOL`; `app/store.py:239 ConversationStore`, `:514 STORE`; `app/config.py` — authoritative env defaults.
- Config: `pyproject.toml` (all: deps, `tool.uv`, `tool.ruff*`, `tool.ty.*`, `tool.pytest.ini_options`), `uv.lock`, `.python-version` (`3.11`), `requirements.txt` (4 runtime lines), `.env.example` (template), `app/config.py`, `.gitignore` (secrets + `data/` + `server.log`).
- Run/docs: `run.sh`, `restart.sh`, `README.md` (sole doc: endpoints, accounts, hash-chain, files/images, limits), `accounts.txt` (secret pool, gitignored), `data/conversations.db*` (runtime SQLite).

## Runtime/Tooling Preferences

- Python `3.11` only (`.python-version`, `requires-python>=3.11`, ruff `py311`, `ty python-version=3.11`). No Node/Bun/npm, no `package.json`/`tsconfig`, no Docker/CI/Makefile.
- Package manager: `uv>=0.12.9` preferred (`uv venv .venv` + `uv pip install -r requirements.txt`); pip-venv fallback in `run.sh`/`restart.sh`. Lockfile `uv.lock`; `[tool.uv] package=false`.
- Runtime deps (4): `fastapi`, `uvicorn[standard]`, `curl_cffi>=0.7`, `pillow`. Dev group: `ty`, `ruff`, `pytest`, `typing-extensions`.
- Env loader `app/config.py:_load_dotenv`: bare `KEY=VALUE` only, no inline-comment stripping, `os.environ.setdefault` (real env wins). Never commit `.env`/`accounts.txt`/`data/`/`server.log`.

## Testing & QA

- Framework: `pytest>=8` (locked `9.1.1`) running `unittest.TestCase` subclasses — both `pytest` and `python -m unittest` work. No `pytest-asyncio`/`anyio`/`respx`/`httpx TestClient`; no `conftest.py`/`fixtures/`; each module ends with `unittest.main()`. Only pytest feature used: `pytest.raises(..., match=)`.
- Layout: `tests/test_keepalive.py` (~20 tests, JWT + `accounts.txt` builders), `test_store.py` (5 tests, `tmpdir` SQLite), `test_branch_race.py` (8 tests, branch-race + JSX citations), `test_image_refusal_failover.py` (7 tests), `test_upstream_error.py` (4 tests). Each opens with regression root-cause docstring; `SCREAMING_SNAKE` hoists magic values (`RACE_EVENTS`, `REFUSAL_MARKER`).
- Pattern: `class FakeAccount(AccountSession)` replays canned SSE dicts via `stream_conversation()` + `pool.register(...)`; sparse `unittest.mock.patch` on seams (`acct.http`, `refresh_access_token`, `upload_image`); isolation via `tempfile.mkdtemp()` + `config.ACCOUNTS_FILE` redirect + `ConversationStore(db_path=tmp)`. Assert failover counts (`total_requests==1`/account) and refusal bytes never stream.
- Coverage: none — no `coverage`/`pytest-cov`, no thresholds, no CI gates. QA gates are `ruff` + `ty` only.
