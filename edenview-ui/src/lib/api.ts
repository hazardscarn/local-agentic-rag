import type {
  AgenticStatusEvent,
  Bbox,
  ChatResponse,
  ChatSessionDetail,
  ChatSessionRecord,
  ClearStaleJobsResponse,
  CollectionRecord,
  DBRecord,
  DocumentRecord,
  IngestAccepted,
  IngestionJobRecord,
  ModelSettings,
  OllamaModelCapabilities,
  PerformanceSettings,
  PreviewResponse,
  RetrievalHit,
  SystemSpecs,
  UnloadAllModelsResponse,
  UpdateModelSettingsResponse,
  WorkspaceSettings,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      // response wasn't JSON -- keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// --- DBs & Collections ---

export const listDbs = () => request<DBRecord[]>("/dbs");

export const createDb = (name: string) =>
  request<DBRecord>("/dbs", { method: "POST", body: JSON.stringify({ name }) });

export const deleteDb = (dbId: string) => request<void>(`/dbs/${dbId}`, { method: "DELETE" });

export const listCollections = (dbName?: string) =>
  request<CollectionRecord[]>(`/collections${dbName ? `?db_name=${encodeURIComponent(dbName)}` : ""}`);

export const getCollection = (name: string) => request<CollectionRecord>(`/collections/${encodeURIComponent(name)}`);

export const deleteCollection = (name: string) =>
  request<void>(`/collections/${encodeURIComponent(name)}`, { method: "DELETE" });

export const listCollectionDocuments = (name: string) =>
  request<DocumentRecord[]>(`/collections/${encodeURIComponent(name)}/documents`);

export const previewCollection = (name: string, limit = 20, offset?: string | null) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (offset) params.set("offset", offset);
  return request<PreviewResponse>(`/collections/${encodeURIComponent(name)}/preview?${params}`);
};

export const listChunkingStrategies = () => request<string[]>("/chunking/strategies");

// --- Ingest ---

export const ingestFile = (form: FormData) =>
  request<IngestAccepted>("/ingest", { method: "POST", body: form });

export const getJob = (jobId: string) => request<IngestionJobRecord>(`/jobs/${jobId}`);

export type JobStatusFilter = "active" | "done" | "error";

// Server-backed job history -- every job regardless of which browser/device kicked
// it off, most-recently-created first. `filename`, if given, searches by that
// substring instead of just returning the most recent `limit`. `status`, if given,
// restricts to "active" (queued or running), "done", or "error".
export const listJobs = (limit = 20, filename?: string, status?: JobStatusFilter) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (filename) params.set("filename", filename);
  if (status) params.set("status", status);
  return request<IngestionJobRecord[]>(`/jobs?${params}`);
};

// Requeues a failed job from its preserved original file -- only works for a job that
// failed after extraction (see pipeline.prepare_retry()); a 400 with a explanatory
// message comes back otherwise (e.g. non-PDF source, or failed before extraction).
export const retryJob = (jobId: string) => request<IngestAccepted>(`/jobs/${jobId}/retry`, { method: "POST" });

// Signals a queued/running job to stop at its next checkpoint -- cooperative, not
// instant (see pipeline.request_cancel()'s docstring): a job mid-extraction stops as
// soon as that call returns, not immediately. A 409 comes back if the job already
// finished, or if it's orphaned (its row says active but this backend process isn't
// actually the one running it, e.g. after a restart).
export const cancelJob = (jobId: string) => request<void>(`/jobs/${jobId}/cancel`, { method: "POST" });

// --- Search & Chat ---

export interface SearchScope {
  query: string;
  db_name?: string;
  collection_names?: string[];
  top_k?: number;
  use_reranker?: boolean;
  file_hashes?: string[];
  strategy?: string;
}

export const runSearch = (body: SearchScope) =>
  request<RetrievalHit[]>("/search", { method: "POST", body: JSON.stringify(body) });

export const runChat = (
  body: SearchScope & { chat_model?: string; session_id?: string; agentic?: boolean },
) => request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) });

// Shared by runChatStream (POST, a fresh turn) and reattachChatStream (GET, a
// still-running one) -- both talk to the same underlying SSE shape (see
// api/routers/chat.py's _subscribe(), which backs both routes). Native
// fetch+ReadableStream (not EventSource, which can't send a POST body), parsing
// the `data: {json}\n\n` frames by hand. Returns null on a "not_running" event
// (only ever sent to a reattaching GET -- see reattachChatStream below) rather
// than throwing, since that's an expected, valid outcome there, not an error.
async function _consumeChatStream(
  res: Response,
  onStatus: (event: AgenticStatusEvent) => void,
  onThinking?: (chunk: string) => void,
): Promise<ChatResponse | null> {
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail ?? errBody);
    } catch {
      // response wasn't JSON -- keep statusText
    }
    throw new ApiError(res.status, detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatResponse | null = null;
  let notRunning = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // last element may be an incomplete frame -- keep it for next read
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const event = JSON.parse(line.slice("data:".length).trim());
      if (event.type === "status") {
        onStatus({
          node: event.node,
          phase: event.phase,
          message: event.message,
          duration_s: event.duration_s,
          query: event.query,
          subquestion_index: event.subquestion_index,
          subquestion_total: event.subquestion_total,
          subquestion_text: event.subquestion_text,
        } satisfies AgenticStatusEvent);
      } else if (event.type === "thinking") {
        onThinking?.(event.message as string);
      } else if (event.type === "result") {
        result = {
          answer: event.answer,
          citations: event.citations,
          model_used: event.model_used,
          session_id: event.session_id,
          thinking: event.thinking,
        };
      } else if (event.type === "not_running") {
        notRunning = true;
      }
    }
  }

  if (notRunning) return null;
  return result;
}

// Agentic-only SSE variant of runChat -- POST /chat/stream forwards every node/tool's
// live status ("Searching your documents...", "Checking whether the answer is
// complete...") as the ADK pipeline runs, since a real turn can genuinely take
// several minutes and a plain spinner reads as broken for that long. `onStatus`
// receives the full AgenticStatusEvent (node/phase/message/duration_s), not just a
// display string -- granular down to individual tool calls, which is what lets the
// flowchart-style status UI light up the active node, not only show a simple
// "current status line". `onThinking`, if given, is called with each chunk of the
// agent's own reasoning narration as it arrives (see runtime.py::run_turn_stream's
// "thinking" event) -- distinct from onStatus, this is the model's actual internal
// narration text, meant to be shown collapsed/expandable, never as the displayed
// answer itself.
export async function runChatStream(
  body: SearchScope & { chat_model?: string; session_id?: string },
  onStatus: (event: AgenticStatusEvent) => void,
  onThinking?: (chunk: string) => void,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, agentic: true }),
  });
  const result = await _consumeChatStream(res, onStatus, onThinking);
  if (!result) throw new Error("Stream ended without a result event");
  return result;
}

// Reattaches to a still-running agentic turn on this session (see
// api/routers/chat.py's GET /chat/stream/{session_id}) -- replays every event
// emitted so far (so the pipeline panel/trace catches up instantly), then
// continues live exactly like a fresh runChatStream() call. Returns null if
// nothing was actually in flight (already finished, never started one, or its
// ~30s grace period already elapsed) -- that's an expected outcome, not an
// error: the caller's job is then just to leave the already-persisted messages
// as they are. `signal` lets a caller abort if the user navigates away again
// before this resolves -- the server-side turn is unaffected either way, since
// it runs independently of any particular subscriber.
export async function reattachChatStream(
  sessionId: string,
  onStatus: (event: AgenticStatusEvent) => void,
  onThinking?: (chunk: string) => void,
  signal?: AbortSignal,
): Promise<ChatResponse | null> {
  const res = await fetch(`${BASE_URL}/chat/stream/${sessionId}`, { signal });
  return _consumeChatStream(res, onStatus, onThinking);
}

export const listChatSessions = (limit = 10, offset = 0) =>
  request<ChatSessionRecord[]>(`/chat/sessions?${new URLSearchParams({ limit: String(limit), offset: String(offset) })}`);

export const getChatSession = (sessionId: string) => request<ChatSessionDetail>(`/chat/sessions/${sessionId}`);

export const deleteChatSession = (sessionId: string) =>
  request<void>(`/chat/sessions/${sessionId}`, { method: "DELETE" });

// --- System & Config ---

export const getSystemInfo = () => request<SystemSpecs>("/system/info");

// Separate from getSystemInfo() -- that's polled every few seconds elsewhere in the
// app (sidebar, system monitor); capabilities are heavier to compute (one extra
// `ollama show` call per pulled model) and static until a model is pulled/removed,
// so this is fetched once per Settings page load instead. Used to filter the
// agent-model/vision-model dropdowns to only options that will actually work.
export const getOllamaModelsWithCapabilities = () => request<OllamaModelCapabilities[]>("/system/ollama/models");

export const unloadOllamaModel = (model: string) =>
  request<void>("/system/ollama/unload", { method: "POST", body: JSON.stringify({ model }) });

// Bulk reset -- evicts every currently-loaded Ollama model (embedding, chat,
// contextual chunking, picture description, agent) in one call to free RAM/VRAM.
export const unloadAllOllamaModels = () =>
  request<UnloadAllModelsResponse>("/system/ollama/unload-all", { method: "POST" });

// Marks any ingestion job left "queued"/"running" by a crashed/restarted backend as
// "error" -- see api/routers/system.py's clear_stale_jobs(). Only touches job status
// rows, never real documents/collections/chat data.
export const clearStaleJobs = () => request<ClearStaleJobsResponse>("/system/jobs/clear-stale", { method: "POST" });

export const getModelSettings = () => request<ModelSettings>("/system/config");

export const updateModelSettings = (updates: Partial<Omit<ModelSettings, "dense_embedding_dim">> & { dense_embedding_dim?: number }) =>
  request<UpdateModelSettingsResponse>("/system/config", { method: "PUT", body: JSON.stringify(updates) });

// The folder everything Edenview builds locally (Qdrant store, DuckDB catalog,
// documents) lives under -- see config.yaml's `workspace:` comment. Changing it
// always requires an API server restart to take effect (cached DuckDB connection/
// Qdrant client singletons) and never moves existing data.
export const getWorkspaceSettings = () => request<WorkspaceSettings>("/system/workspace");

export const updateWorkspace = (root: string) =>
  request<WorkspaceSettings>("/system/workspace", { method: "PUT", body: JSON.stringify({ root }) });

// Opens a native OS folder-picker dialog on the machine running the backend --
// viable since this is a local single-user app (server and browser share a machine).
// Returns null if the user canceled the dialog.
export const browseWorkspaceFolder = () =>
  request<{ path: string | null }>("/system/workspace/browse", { method: "POST" }).then((r) => r.path);

export const getPerformanceSettings = () => request<PerformanceSettings>("/system/performance");

// Pass `null` for a field to revert it to auto-detecting; omit a field to leave it
// as whatever it already was.
export const updatePerformanceSettings = (updates: {
  num_threads?: number | null;
  page_batch_size?: number | null;
  max_concurrent_extractions?: number | null;
}) => request<PerformanceSettings>("/system/performance", { method: "PUT", body: JSON.stringify(updates) });

// --- Files & documents ---

export const fileUrl = (path: string) => `${BASE_URL}/files?${new URLSearchParams({ path })}`;

// Renders one page of a previously-ingested PDF, with an optional highlight box drawn
// server-side (pypdfium2) -- see api/routers/documents.py. `bbox` is the same
// normalized [l, t, r, b] tuple already on a RetrievalHit/PreviewChunk.
export const documentPageUrl = (fileHash: string, pageNo: number, bbox?: Bbox | null) => {
  const params = bbox ? `?${new URLSearchParams({ bbox: bbox.join(",") })}` : "";
  return `${BASE_URL}/documents/${fileHash}/pages/${pageNo}${params}`;
};
