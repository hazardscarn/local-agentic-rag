# Edenview Portal UI

Next.js frontend for the Edenview API (`../api`) — Ingestion, Collections, Chat, and
Settings views. Talks to the FastAPI backend over plain `fetch` (see `src/lib/api.ts`);
no server-side data fetching of its own, so it's a thin client over the API described
in `../api/README.md`.

## Running

The backend must already be running first (from the project root):

```bash
PYTHONPATH=. venv/Scripts/python.exe -m uvicorn api.app:app --reload --port 8000
```

Then, from this directory:

```bash
npm install
npm run dev
```

Open http://localhost:3000 — it redirects to `/ingestion`.

`.env.local` sets `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000` in
code if unset, so `.env.local` is optional unless the backend runs elsewhere).

## Views

- **`/ingestion`** — upload a file, pick a chunking strategy, track the ingest job
  through extraction/chunking/embedding, preview the resulting chunks. Job history is
  tracked client-side only (`localStorage`) — there's no `GET /jobs` list endpoint on
  the backend, only `GET /jobs/{id}`.
- **`/collections`** — the DuckDB-backed catalog: every database, its collections,
  and drill-down into documents + chunk preview.
- **`/chat`** — scope a question to a database or specific collections, get an answer
  with numbered citations from `POST /chat` (a single retrieval pass + one LLM call,
  not an agentic loop).
- **`/settings`** — model configuration read from and written back to `config.yaml`
  via `GET`/`PUT /system/config`. Each field is labeled whether it applies
  immediately or needs an API server restart (see `api/routers/config.py`'s
  `RESTART_REQUIRED_KEYS` for why that split exists).

## Stack notes

Next.js 16 (App Router, Turbopack) + Tailwind v4 + shadcn/ui — note shadcn here is
built on **Base UI**, not Radix (no `asChild`, use the `render` prop instead; see
`src/components/ui/*.tsx` for the actual primitives in use before assuming Radix-era
APIs). Data fetching/caching via TanStack Query. No global state library — page-level
`useState` plus React Query's cache is enough for this app's size.
