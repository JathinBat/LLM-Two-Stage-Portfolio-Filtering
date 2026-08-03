# API keys & secrets — setup

All API keys for this project now live in a **single file: `.env`** at the repo root.
Nothing is hardcoded in the source anymore, and `.env` is gitignored so it never gets committed.

## How it works
- `.env` — your real keys (gitignored, never committed). One `KEY=value` per line.
- `.env.example` — a committed template with blank values, so anyone cloning the repo knows which keys are needed.
- `env_loader.py` — a tiny helper that reads `.env` into environment variables. The main tools also have an inline loader, so `.env` is picked up automatically no matter which folder you run from.
- The code reads every key via `os.environ.get("...")`, so keys come from `.env` (or from real environment variables you set in the shell, which take precedence).

## Keys used
`OPENAI_API_KEY`, `NYT_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `GNEWS_API_KEY`, `NEWSAPI_ORG_KEY`, `THENEWSAPI_KEY`

## First-time setup on a new machine
1. Copy `.env.example` to `.env`.
2. Paste your keys into `.env`.
3. Run any tool as usual — the key is loaded automatically.

## IMPORTANT — rotate the old keys
The keys that were previously hardcoded are still in this repo's **git history** and were likely exposed. Before making the repo public, **revoke/rotate every key at its provider** and put the fresh values in `.env`:
- OpenAI: https://platform.openai.com/api-keys
- NY Times: https://developer.nytimes.com
- Alpha Vantage / GNews / NewsAPI.org / TheNewsAPI: regenerate in each account.

If you're starting a brand-new GitHub repo from this folder (fresh history), the old keys won't travel with it — but the copies already pushed/committed elsewhere still need rotating.
