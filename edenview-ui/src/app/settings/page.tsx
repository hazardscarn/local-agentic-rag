"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  browseWorkspaceFolder,
  clearStaleJobs,
  getModelSettings,
  getOllamaModelsWithCapabilities,
  getPerformanceSettings,
  getSystemInfo,
  getWorkspaceSettings,
  unloadAllOllamaModels,
  updateModelSettings,
  updatePerformanceSettings,
  updateWorkspace,
} from "@/lib/api";
import type { ModelSettings } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { ModelField } from "@/components/settings/model-field";
import { AgentModelSelect } from "@/components/settings/agent-model-select";
import { FieldHelp } from "@/components/shared/field-help";
import { Loader2, Cpu, MemoryStick, MonitorCog, TriangleAlert, FolderCog, FolderOpen, Gauge, Bot, Eraser } from "lucide-react";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: system } = useQuery({ queryKey: ["system-info"], queryFn: getSystemInfo });
  const { data: settings, isLoading } = useQuery({ queryKey: ["model-settings"], queryFn: getModelSettings });
  const { data: workspace } = useQuery({ queryKey: ["workspace-settings"], queryFn: getWorkspaceSettings });
  const { data: performance } = useQuery({ queryKey: ["performance-settings"], queryFn: getPerformanceSettings });
  // Separate from system-info -- see getOllamaModelsWithCapabilities()'s own comment
  // for why this isn't folded into the frequently-polled GET /system/info instead.
  const { data: capModels } = useQuery({ queryKey: ["ollama-models-capabilities"], queryFn: getOllamaModelsWithCapabilities });

  const [form, setForm] = useState<ModelSettings | null>(null);
  const [workspaceRoot, setWorkspaceRoot] = useState("");
  // Empty string = "auto" (no override) -- pre-filled with the current override's
  // value, if any, otherwise left blank so the auto-detected value shows as a
  // placeholder instead of a value the user would have to notice and clear.
  const [numThreadsInput, setNumThreadsInput] = useState("");
  const [pageBatchSizeInput, setPageBatchSizeInput] = useState("");
  const [maxConcurrentInput, setMaxConcurrentInput] = useState("");

  useEffect(() => {
    if (settings) setForm(settings);
  }, [settings]);

  useEffect(() => {
    if (workspace) setWorkspaceRoot(workspace.root);
  }, [workspace]);

  useEffect(() => {
    if (performance) {
      setNumThreadsInput(performance.num_threads_is_auto ? "" : String(performance.num_threads));
      setPageBatchSizeInput(performance.page_batch_size_is_auto ? "" : String(performance.page_batch_size));
      setMaxConcurrentInput(performance.max_concurrent_extractions_is_auto ? "" : String(performance.max_concurrent_extractions));
    }
  }, [performance]);

  const mutation = useMutation({
    mutationFn: updateModelSettings,
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["model-settings"] });
      if (res.restart_required.length > 0) {
        toast.warning(`Saved. Restart the API server to apply: ${res.restart_required.join(", ")}`);
      } else {
        toast.success("Saved — changes are live now.");
      }
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const workspaceMutation = useMutation({
    mutationFn: updateWorkspace,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspace-settings"] });
      toast.warning("Saved. Restart the API server to apply — existing data was not moved.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const browseMutation = useMutation({
    mutationFn: browseWorkspaceFolder,
    onSuccess: (path) => {
      if (path) setWorkspaceRoot(path);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const performanceMutation = useMutation({
    mutationFn: updatePerformanceSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["performance-settings"] });
      toast.warning("Saved. Extraction threads/page batch size apply to the next ingestion — max concurrent extractions needs an API server restart.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Marks any ingestion job left "queued"/"running" by a crashed/restarted backend
  // as "error" -- only touches job status rows, never real documents/collections/
  // chat data. Shares its backend logic with scripts/fresh_start.py.
  const clearStaleJobsMutation = useMutation({
    mutationFn: clearStaleJobs,
    onSuccess: (res) => {
      if (res.cleared_count === 0) {
        toast.success("No stale jobs found — already clean.");
        return;
      }
      toast.success(
        `Cleared ${res.cleared_count} stale job${res.cleared_count > 1 ? "s" : ""}: ${res.cleared_filenames.join(", ")}`,
      );
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Evicts every Ollama model currently loaded (embedding, chat, contextual chunking,
  // picture description, agent) to free RAM/VRAM in one call -- the bulk counterpart
  // to unloading one model at a time from the sidebar's system monitor. Doesn't
  // interrupt a call actively streaming right now (Ollama has no cancel endpoint) --
  // it only reclaims memory a model is holding idle.
  const unloadAllMutation = useMutation({
    mutationFn: unloadAllOllamaModels,
    onSuccess: (res) => {
      if (res.models_unloaded.length === 0) {
        toast.success("No models were loaded — already clean.");
        return;
      }
      toast.success(`Unloaded ${res.models_unloaded.length} model${res.models_unloaded.length > 1 ? "s" : ""}: ${res.models_unloaded.join(", ")}`);
      queryClient.invalidateQueries({ queryKey: ["system-live"] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (isLoading || !form) {
    return (
      <div className="flex items-center gap-2 px-8 py-10 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading settings…
      </div>
    );
  }

  const set = <K extends keyof ModelSettings>(key: K, value: ModelSettings[K]) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const diff: Partial<ModelSettings> = {};
  if (settings) {
    (Object.keys(form) as (keyof ModelSettings)[]).forEach((key) => {
      if (form[key] !== settings[key]) (diff as Record<string, unknown>)[key] = form[key];
    });
  }
  const hasChanges = Object.keys(diff).length > 0;

  const pulled = system?.ollama.models ?? [];

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-8 py-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">Model configuration read from and saved back to config.yaml.</p>
      </div>

      {system && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">This machine</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <InfoStat icon={Cpu} label="CPU" value={`${system.cpu_cores_physical ?? "?"} cores`} />
              <InfoStat icon={MemoryStick} label="RAM" value={`${system.ram_available_gb} / ${system.ram_total_gb} GB free`} />
              <InfoStat
                icon={MonitorCog}
                label="GPU"
                value={system.gpus[0] ? `${system.gpus[0].name}${system.gpus[0].vram_total_mb ? ` (${system.gpus[0].vram_total_mb}MB)` : ""}` : "None detected"}
              />
              <InfoStat
                icon={Gauge}
                label="Extraction acceleration"
                value={
                  system.torch_acceleration.device === "cuda"
                    ? `CUDA (${system.torch_acceleration.gpu_name ?? "GPU"})`
                    : system.torch_acceleration.device === "mps"
                      ? "Apple Silicon (MPS)"
                      : "CPU only"
                }
              />
            </div>
            {system.gpus.length > 0 && system.torch_acceleration.device !== "cuda" && (
              <Alert variant="destructive">
                <TriangleAlert className="size-4" />
                <AlertTitle>GPU detected, but extraction isn&apos;t using it</AlertTitle>
                <AlertDescription>
                  Docling&apos;s extraction models (layout analysis, OCR, table structure, picture
                  classification) run on the CPU right now — the installed torch build has no CUDA
                  support, even though a GPU is present. Run{" "}
                  <code className="rounded bg-muted px-1 py-0.5">python scripts/install_torch.py</code>{" "}
                  from the project root (one-time setup step) to detect this GPU and install a matching
                  CUDA build — commonly several times faster for extraction.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <FolderCog className="size-4" /> Workspace
          </CardTitle>
          <CardDescription>Where every database, collection, document, and chat session is stored on disk.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Alert variant="destructive">
            <TriangleAlert className="size-4" />
            <AlertTitle>Set this once, right after installing</AlertTitle>
            <AlertDescription>
              Everything you&apos;ve ingested — every collection, document, and chat — lives under this
              folder. Changing it here does not move that data; it just points a fresh process at a
              different folder, making existing work look like it disappeared until you point it back.
              Keep it on a local, always-on disk. A network drive or a removable drive that isn&apos;t
              always connected will make the vector store and catalog unreliable (stalls, lock errors,
              or corruption) — neither is built to run over a network.
            </AlertDescription>
          </Alert>

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <Label>Workspace folder</Label>
              <FieldHelp>
                Relative paths (e.g. &quot;edenview_data&quot;) resolve against the project root; an
                absolute path (e.g. &quot;D:\EdenviewData&quot; or &quot;/home/me/edenview_data&quot;)
                is used as-is. Always requires an API server restart to take effect.
              </FieldHelp>
            </div>
            <div className="flex gap-2">
              <Input
                value={workspaceRoot}
                onChange={(e) => setWorkspaceRoot(e.target.value)}
                placeholder="edenview_data"
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                disabled={browseMutation.isPending}
                onClick={() => browseMutation.mutate()}
              >
                {browseMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <FolderOpen className="size-4" />}
                Browse…
              </Button>
            </div>
            {workspace && <p className="text-xs text-muted-foreground">Currently resolves to: {workspace.resolved_path}</p>}
          </div>

          <div className="flex justify-end">
            <Button
              disabled={!workspaceRoot.trim() || workspaceRoot === workspace?.root || workspaceMutation.isPending}
              onClick={() => workspaceMutation.mutate(workspaceRoot.trim())}
            >
              {workspaceMutation.isPending && <Loader2 className="size-4 animate-spin" />}
              Save workspace folder
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Gauge className="size-4" /> Performance
          </CardTitle>
          <CardDescription>
            How Docling's extraction pipeline uses this machine's resources. Both auto-detected by
            default — leave blank unless you want to override.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5">
                <Label>Extraction threads</Label>
                <FieldHelp>
                  How many threads Docling uses per document extraction. Auto-detected as this
                  machine&apos;s CPU count minus 2 (leaving headroom for the OS and everything else
                  running), never below 1. Set a number to override, or clear the field to go back
                  to auto-detecting.
                </FieldHelp>
              </div>
              <Input
                type="number"
                min={1}
                value={numThreadsInput}
                onChange={(e) => setNumThreadsInput(e.target.value)}
                placeholder={performance ? `auto (${performance.num_threads})` : "auto"}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5">
                <Label>Page batch size</Label>
                <FieldHelp>
                  How many pages of one document Docling processes in a batch internally. Docling&apos;s
                  own default is 4 — a higher number can mean faster throughput at the cost of more
                  memory held at once. Applies process-wide (every extraction in this run), not just
                  the next one. Clear the field to go back to Docling&apos;s default.
                </FieldHelp>
              </div>
              <Input
                type="number"
                min={1}
                value={pageBatchSizeInput}
                onChange={(e) => setPageBatchSizeInput(e.target.value)}
                placeholder={performance ? `auto (${performance.page_batch_size})` : "auto"}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5">
                <Label>Max concurrent extractions</Label>
                <FieldHelp>
                  How many documents can run extraction (the actual resource-hungry step — model
                  loading + inference) at the same time. Ingesting more files than this just queues
                  the rest rather than running everything at once. Default 4 — more cores doesn&apos;t
                  mean more concurrent extractions are safe, since RAM and a shared GPU (if present)
                  are the real limits, and firing too many at once makes the whole batch slower, not
                  faster. Requires an API server restart to take effect.
                </FieldHelp>
              </div>
              <Input
                type="number"
                min={1}
                value={maxConcurrentInput}
                onChange={(e) => setMaxConcurrentInput(e.target.value)}
                placeholder={performance ? `auto (${performance.max_concurrent_extractions})` : "auto"}
              />
            </div>
          </div>
          <div className="flex justify-end">
            <Button
              disabled={performanceMutation.isPending}
              onClick={() =>
                performanceMutation.mutate({
                  num_threads: numThreadsInput.trim() ? Number(numThreadsInput) : null,
                  page_batch_size: pageBatchSizeInput.trim() ? Number(pageBatchSizeInput) : null,
                  max_concurrent_extractions: maxConcurrentInput.trim() ? Number(maxConcurrentInput) : null,
                })
              }
            >
              {performanceMutation.isPending && <Loader2 className="size-4 animate-spin" />}
              Save performance settings
            </Button>
          </div>
        </CardContent>
      </Card>

      {system && (!system.ollama.available || system.ollama.models.length === 0) && (
        <Alert variant="destructive">
          <TriangleAlert className="size-4" />
          <AlertTitle>{system.ollama.available ? "No models pulled yet" : "Ollama unreachable"}</AlertTitle>
          <AlertDescription>
            {system.ollama.available
              ? "Pull the models you need before selecting them below, e.g. `ollama pull bge-m3` and `ollama pull qwen3:4b`."
              : `Couldn't reach Ollama at ${system.ollama.host}. Make sure it's running before picking models here.`}
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Embedding & retrieval</CardTitle>
          <CardDescription>Changing the embedding model without re-ingesting existing collections will break them — no auto-migration.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <ModelField settingKey="dense_embedding" label="Dense embedding model" value={form.dense_embedding} onChange={(v) => set("dense_embedding", v)} pulledModels={pulled} />
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <Label>Dense embedding dimension</Label>
              <FieldHelp>
                Not independently editable — a model's output dimension is intrinsic to that model, not a
                free choice. Editing this separately from the embedding model risks a mismatch: either
                ingestion fails loudly (Qdrant rejects the wrong vector size), or worse, if two different
                models happen to share a dimension, embeddings silently mix incompatible vector spaces with
                no error at all. Saving a new embedding model above re-detects this automatically by
                calling it once and measuring its actual output length.
              </FieldHelp>
            </div>
            <p className="flex h-8 items-center rounded-lg border border-dashed border-input px-2.5 text-sm text-muted-foreground">
              {form.dense_embedding_dim} (detected from {form.dense_embedding})
            </p>
          </div>
          <ModelField settingKey="sparse_embedding" label="Sparse (BM25) model" value={form.sparse_embedding} onChange={(v) => set("sparse_embedding", v)} />
          <ModelField settingKey="reranker" label="Reranker model" value={form.reranker} onChange={(v) => set("reranker", v)} />
          <ModelField settingKey="tokenizer" label="Tokenizer" value={form.tokenizer} onChange={(v) => set("tokenizer", v)} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">LLMs</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <ModelField settingKey="chat_llm" label="Chat / answer model" value={form.chat_llm} onChange={(v) => set("chat_llm", v)} pulledModels={pulled} />
          <ModelField settingKey="contextual_llm" label="Contextual chunking model" value={form.contextual_llm} onChange={(v) => set("contextual_llm", v)} pulledModels={pulled} />
          <ModelField settingKey="picture_description_llm" label="Picture description model" value={form.picture_description_llm} onChange={(v) => set("picture_description_llm", v)} pulledModels={pulled} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Bot className="size-4" /> Agentic RAG
          </CardTitle>
          <CardDescription>
            The model driving the agentic pipeline&apos;s reword/search/eval/deep-search loop -- separate
            from the chat model above, since Simple RAG and Agentic RAG are independently configurable.
            Different hardware can run different models here; the dropdowns only list pulled models that
            actually report the capability each field needs.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <AgentModelSelect
            settingKey="agent_model"
            label="Agent model (must support tool-calling)"
            capability="tools"
            value={form.agent_model}
            onChange={(v) => set("agent_model", v)}
            models={capModels ?? []}
          />
          <AgentModelSelect
            settingKey="agent_vision_model"
            label="Agent vision model"
            capability="vision"
            value={form.agent_vision_model ?? ""}
            onChange={(v) => set("agent_vision_model", v || null)}
            models={capModels ?? []}
            allowUnset
            unsetLabel="Use agent model's vision, if supported"
          />
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Label>Max refinement iterations</Label>
              <Badge variant="secondary" className="text-[10px]">restart required</Badge>
            </div>
            <Input
              type="number"
              min={1}
              value={form.agent_max_iterations}
              onChange={(e) => set("agent_max_iterations", Number(e.target.value) || 1)}
            />
            <p className="text-xs text-muted-foreground">
              How many reword/search/eval passes one sub-question&apos;s research loop can take before
              giving up. The loop usually exits early once the eval step says the findings are enough --
              this is just the ceiling.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Connection</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <ModelField settingKey="ollama_host" label="Ollama host" value={form.ollama_host ?? ""} onChange={(v) => set("ollama_host", v)} />
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <Label>Model idle timeout (keep_alive)</Label>
              <FieldHelp>
                How long a model stays loaded in RAM/VRAM after its last use before Ollama evicts it and the
                next call has to reload it from disk — an idle timer that resets on every call, not a fixed
                schedule. Accepts Ollama&apos;s own duration syntax: &quot;30m&quot;, &quot;1h&quot;, &quot;-1&quot;
                (never auto-evict), or &quot;0&quot; (unload immediately — same as the Unload button in the
                sidebar). Applies to every model this app calls (embedding, chat, contextual chunking, picture
                description) and takes effect immediately, no restart needed.
              </FieldHelp>
            </div>
            <Input
              value={form.ollama_keep_alive ?? ""}
              onChange={(e) => set("ollama_keep_alive", e.target.value)}
              placeholder="30m"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Eraser className="size-4" /> Maintenance
          </CardTitle>
          <CardDescription>
            Clears any ingestion job left &quot;queued&quot;/&quot;running&quot; by a backend that crashed or
            was restarted mid-job — nothing else ever marks those rows finished once their process is gone.
            Only touches job status rows, never real documents/collections/chat data.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Button
            variant="secondary"
            disabled={clearStaleJobsMutation.isPending}
            onClick={() => clearStaleJobsMutation.mutate()}
            className="w-fit"
          >
            {clearStaleJobsMutation.isPending && <Loader2 className="size-4 animate-spin" />}
            Clear stale ingestion jobs
          </Button>

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <Button
                variant="secondary"
                disabled={unloadAllMutation.isPending}
                onClick={() => unloadAllMutation.mutate()}
                className="w-fit"
              >
                {unloadAllMutation.isPending && <Loader2 className="size-4 animate-spin" />}
                Unload all Ollama models (free VRAM)
              </Button>
              <FieldHelp>
                Evicts every currently-loaded Ollama model — embedding, chat, contextual chunking, picture
                description, agent — from RAM/VRAM right away, instead of waiting on each one&apos;s idle
                timeout. Doesn&apos;t interrupt a call that&apos;s actively generating right now (Ollama has
                no way to cancel one mid-flight) — that finishes on its own; this only frees memory models
                are holding while idle.
              </FieldHelp>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button disabled={!hasChanges || mutation.isPending} onClick={() => mutation.mutate(diff)}>
          {mutation.isPending && <Loader2 className="size-4 animate-spin" />}
          Save changes
        </Button>
      </div>
    </div>
  );
}

function InfoStat({ icon: Icon, label, value }: { icon: typeof Cpu; label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <Icon className="mt-0.5 size-4 text-muted-foreground" />
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-sm">{value}</span>
      </div>
    </div>
  );
}
