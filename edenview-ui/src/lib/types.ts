// Mirrors the Pydantic models in api/schemas.py, edenview_ingestion/catalog/models.py,
// edenview_RAG/retrieval/models.py, and edenview_ingestion/system_inspector.py.
// Hand-typed rather than codegen'd -- the API surface is small and stable enough that
// this is less overhead than wiring an OpenAPI generator.

export interface DBRecord {
  db_id: string;
  name: string;
  created_at: string;
}

export interface CollectionRecord {
  collection_id: string;
  db_id: string;
  qdrant_collection_name: string;
  chunking_strategy: string;
  embedding_model: string;
  dense_dim: number;
  sparse_model: string | null;
  status: string;
  chunk_count: number;
  doc_count: number;
  created_at: string;
}

export interface DocumentRecord {
  doc_id: string;
  file_hash: string;
  filename: string;
  source_path: string | null;
  input_format: string | null;
  num_pages: number | null;
  first_ingested_at: string;
}

export interface IngestionJobRecord {
  job_id: string;
  collection_id: string;
  doc_id: string | null;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  // Stored at job-creation time -- known immediately, unlike doc_id.
  filename: string | null;
  stage: "extracting" | "chunking" | "embedding" | null;
  stage_current: number | null;
  stage_total: number | null;
  stage_pct: number | null;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  // Joined in server-side for display -- not stored on the job row itself.
  qdrant_collection_name: string | null;
  db_name: string | null;
}

export interface IngestAccepted {
  job_id: string;
  status: string;
  qdrant_collection_name: string;
}

export interface ChunkImage {
  picture_id: string;
  image_path: string;
  caption: string | null;
  page_no: number | null;
  kind: "picture" | "table";
}

// Normalized (0..1), top-left-origin: [left, top, right, bottom]. Only set for
// HybridChunker-based strategies (hybrid_docling, parent_child, contextual); never
// for recursive_overlap. Powers the visual grounding panel.
export type Bbox = [number, number, number, number];

export interface PreviewChunk {
  chunk_id: string;
  text: string;
  page_no: number | null;
  bbox: Bbox | null;
  kind: string;
  strategy: string;
  file_hash: string;
  images: ChunkImage[];
  // Only set for a parent_child strategy's "child" chunks -- the full parent
  // context this fragment was split from.
  parent_text: string | null;
}

export interface PreviewResponse {
  chunks: PreviewChunk[];
  next_offset: string | null;
}

export interface RetrievalHit {
  chunk_id: string;
  score: number;
  text: string;
  context_text: string;
  collection_name: string;
  strategy: string;
  kind: string;
  page_no: number | null;
  bbox: Bbox | null;
  headings: string[];
  doc_stem: string;
  file_hash: string;
  images: ChunkImage[];
}

export interface ChatResponse {
  answer: string;
  citations: RetrievalHit[];
  model_used: string;
  session_id: string;
  // Agentic mode only -- the agent's own reasoning/planning narration for this
  // turn, kept separate from `answer` so it can be shown as an expandable section.
  thinking?: string | null;
}

// One "status" event from POST /chat/stream (see runtime.py::run_turn_stream and
// callbacks.py's track_agent_start/end + track_tool_start/end). Granular down to
// individual tool calls, not just top-level agent phases -- `node` is an agent name
// (e.g. "reworder", "eval", "deep_search") or a tool name (e.g. "vector_search",
// "get_images"); `phase` is "start" (always has `message`, a human-readable label)
// or "end" (has `duration_s` instead, no `message`). Root-level events forwarded
// from the outer event stream (e.g. "Working (query_pipeline)...") have `message`
// but no `node`/`phase`/`duration_s` -- this shape is intentionally the same one
// a future flowchart-style status UI would consume to light up the active node.
export interface AgenticStatusEvent {
  node?: string;
  phase?: "start" | "end";
  message?: string;
  duration_s?: number | null;
  // For the unified researcher architecture -- shows the actual search query text.
  query?: string;
  subquestion_index?: number;
  subquestion_total?: number;
  subquestion_text?: string;
}

export interface ChatSessionRecord {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRecord {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  citations: RetrievalHit[] | null;
  model_used: string | null;
  created_at: string;
}

export interface ChatSessionDetail {
  session_id: string;
  title: string;
  messages: ChatMessageRecord[];
}

export interface GPUInfo {
  name: string;
  vendor: "nvidia" | "apple" | "unknown";
  vram_total_mb: number | null;
  vram_free_mb: number | null;
  unified_memory: boolean;
}

export interface OllamaModelInfo {
  name: string;
  size_gb: number;
}

export interface OllamaInfo {
  available: boolean;
  host: string;
  models: OllamaModelInfo[];
  error: string | null;
}

export interface LoadedModelInfo {
  name: string;
  size_gb: number;
  size_vram_gb: number;
  expires_at: string | null;
}

export interface TorchAccelerationInfo {
  installed: boolean;
  device: "cuda" | "mps" | "cpu" | null;
  gpu_name: string | null;
}

export interface SystemSpecs {
  platform: string;
  platform_release: string;
  architecture: string;
  cpu_cores_physical: number | null;
  cpu_cores_logical: number | null;
  ram_total_gb: number;
  ram_available_gb: number;
  gpus: GPUInfo[];
  ollama: OllamaInfo;
  loaded_models: LoadedModelInfo[];
  torch_acceleration: TorchAccelerationInfo;
}

export interface ModelSettings {
  tokenizer: string;
  dense_embedding: string;
  dense_embedding_dim: number;
  sparse_embedding: string;
  contextual_llm: string;
  picture_description_llm: string;
  chat_llm: string;
  reranker: string;
  ollama_host: string | null;
  ollama_keep_alive: string | null;
  // Agentic RAG's model config (edenview_RAG/agentic_rag) -- separate from chat_llm
  // since Simple RAG and Agentic RAG are independently configurable. Always
  // restart-required (see RESTART_REQUIRED_KEYS below) -- unlike chat_llm, these are
  // baked into @lru_cache'd LLM singletons and an import-time tool-calling check.
  agent_model: string;
  // null means "reuse agent_model if it's vision-capable, else unavailable" -- see
  // edenview_RAG.agentic_rag.config.get_vision_model()'s fallback logic.
  agent_vision_model: string | null;
}

export interface UpdateModelSettingsResponse {
  updated: ModelSettings;
  restart_required: string[];
}

// One pulled Ollama model's name/size plus its reported capabilities (e.g.
// ["completion", "tools", "vision", "thinking"]) -- from GET /system/ollama/models,
// a separate call from GET /system/info (which is polled every few seconds
// elsewhere in the app and deliberately doesn't carry this heavier-to-compute data).
// Used to filter the agent-model/vision-model dropdowns to only options that will
// actually work.
export interface OllamaModelCapabilities {
  name: string;
  size_gb: number;
  capabilities: string[];
}

// Response of POST /system/jobs/clear-stale -- marks any "queued"/"running"
// ingestion job left behind by a crashed/restarted backend as "error". Only ever
// touches job status rows, never real documents/collections/chat data.
export interface ClearStaleJobsResponse {
  cleared_count: number;
  cleared_filenames: string[];
}

// Bulk counterpart to a single-model unload -- names of every Ollama model that was
// actually loaded and got evicted from RAM/VRAM. Empty if nothing was loaded.
export interface UnloadAllModelsResponse {
  models_unloaded: string[];
}

export interface WorkspaceSettings {
  // As stored in config.yaml -- relative or absolute.
  root: string;
  // Absolute, resolved form -- what's actually in use right now.
  resolved_path: string;
}

export interface PerformanceSettings {
  num_threads: number;
  page_batch_size: number;
  max_concurrent_extractions: number;
  num_threads_is_auto: boolean;
  page_batch_size_is_auto: boolean;
  max_concurrent_extractions_is_auto: boolean;
}

// Keys that only take effect after restarting the API server -- mirrors
// api/routers/config.py's RESTART_REQUIRED_KEYS, but the authoritative value always
// comes back on the PUT response; this is just used to render the badge before any
// save has happened yet.
export const RESTART_REQUIRED_KEYS = new Set([
  "tokenizer",
  "sparse_embedding",
  "contextual_llm",
  "picture_description_llm",
  "reranker",
  "agent_model",
  "agent_vision_model",
]);
