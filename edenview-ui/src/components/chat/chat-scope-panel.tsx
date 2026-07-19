"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getModelSettings, getSystemInfo, listChunkingStrategies, listCollections, listDbs } from "@/lib/api";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { CheckboxGroup } from "@/components/ui/checkbox-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { PanelRightClose, ChevronDown, ChevronRight } from "lucide-react";

export interface ChatScope {
  // Fully resolved qdrant_collection_names -- selecting a whole database (its
  // parent checkbox) just means every one of that database's collection names
  // ends up in here, same as picking them all individually. No separate "db mode":
  // the backend always gets a flat collection_names list, never db_name.
  collectionNames: string[];
  topK: number;
  useReranker: boolean;
  strategy: string;
  chatModel: string;
  // Simple RAG (one retrieval pass + one LLM call, POST /chat) vs. Agentic RAG
  // (edenview_RAG.agentic_rag's ADK-based reword/retrieve/eval/deep-search
  // pipeline, POST /chat/stream) -- one flat pipeline, no effort tiers.
  agentic: boolean;
}

export function ChatScopePanel({
  scope,
  onChange,
  onCollapse,
}: {
  scope: ChatScope;
  onChange: (next: ChatScope) => void;
  onCollapse: () => void;
}) {
  const { data: dbs } = useQuery({ queryKey: ["dbs"], queryFn: listDbs });
  const { data: collections } = useQuery({ queryKey: ["collections", "all"], queryFn: () => listCollections() });
  const { data: strategies } = useQuery({ queryKey: ["strategies"], queryFn: listChunkingStrategies });
  const { data: system } = useQuery({ queryKey: ["system-info"], queryFn: getSystemInfo });
  const { data: modelSettings } = useQuery({ queryKey: ["model-settings"], queryFn: getModelSettings });
  const [expandedDbIds, setExpandedDbIds] = useState<Set<string>>(new Set());

  const set = <K extends keyof ChatScope>(key: K, value: ChatScope[K]) => onChange({ ...scope, [key]: value });

  const toggleExpanded = (dbId: string) =>
    setExpandedDbIds((prev) => {
      const next = new Set(prev);
      if (next.has(dbId)) next.delete(dbId);
      else next.add(dbId);
      return next;
    });

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col gap-5 overflow-y-auto border-l border-border px-4 py-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Chat settings</span>
        <Button variant="ghost" size="icon" className="size-7" onClick={onCollapse} title="Hide chat settings">
          <PanelRightClose className="size-4" />
        </Button>
      </div>

      <div>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Chat mode</h2>
        <div className="flex rounded-lg border border-border p-1">
          <Button
            variant={scope.agentic ? "ghost" : "secondary"}
            size="sm"
            className="flex-1"
            onClick={() => set("agentic", false)}
          >
            Simple RAG
          </Button>
          <Button
            variant={scope.agentic ? "secondary" : "ghost"}
            size="sm"
            className="flex-1"
            onClick={() => set("agentic", true)}
          >
            Agentic RAG
          </Button>
        </div>
        <p className="mt-1.5 text-xs text-muted-foreground">
          {scope.agentic
            ? "Rewords your question, retrieves, evaluates its own findings, and looks deeper before answering. Slower, more thorough."
            : "One retrieval pass + one LLM call."}
        </p>
      </div>

      <Separator />

      <div className="flex flex-col gap-2">
        <Label>Scope</Label>
        <p className="text-xs text-muted-foreground">
          Check a database to search all of it, or expand it to pick specific collections.
        </p>
        <div className="flex max-h-64 flex-col gap-1 overflow-y-auto rounded-lg border border-border p-1.5">
          {dbs?.map((db) => {
            const dbCollections = collections?.filter((c) => c.db_id === db.db_id) ?? [];
            const allNames = dbCollections.map((c) => c.qdrant_collection_name);
            const selectedInDb = scope.collectionNames.filter((n) => allNames.includes(n));
            const isExpanded = expandedDbIds.has(db.db_id);
            return (
              <div key={db.db_id} className="rounded-md">
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => toggleExpanded(db.db_id)}
                    disabled={allNames.length === 0}
                    className="flex size-5 shrink-0 items-center justify-center text-muted-foreground hover:text-foreground disabled:opacity-30"
                  >
                    {allNames.length > 0 &&
                      (isExpanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />)}
                  </button>
                  <CheckboxGroup
                    value={selectedInDb}
                    onValueChange={(names) => {
                      const outsideThisDb = scope.collectionNames.filter((n) => !allNames.includes(n));
                      set("collectionNames", [...outsideThisDb, ...names]);
                    }}
                    allValues={allNames}
                    className="flex-1"
                  >
                    <label className="flex flex-1 items-center gap-2 py-1 text-sm">
                      <Checkbox parent disabled={allNames.length === 0} />
                      <span className="truncate font-medium">{db.name}</span>
                    </label>
                    {isExpanded && (
                      <div className="ml-1 flex flex-col gap-1 border-l border-border py-1 pl-4">
                        {dbCollections.map((c) => (
                          <label key={c.collection_id} className="flex items-center gap-2 text-sm">
                            <Checkbox value={c.qdrant_collection_name} />
                            <span className="truncate text-muted-foreground">{c.qdrant_collection_name}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </CheckboxGroup>
                </div>
              </div>
            );
          })}
          {dbs?.length === 0 && <p className="p-2 text-xs text-muted-foreground">No databases yet.</p>}
        </div>
      </div>

      <Separator />

      <div className="flex flex-col gap-1.5">
        <Label>Top K</Label>
        <Input
          type="number"
          min={1}
          max={20}
          value={scope.topK}
          onChange={(e) => set("topK", Number(e.target.value) || 5)}
        />
      </div>

      <label className="flex items-center gap-2 text-sm">
        <Checkbox checked={scope.useReranker} onCheckedChange={(v) => set("useReranker", v === true)} />
        Use reranker
      </label>

      <div className="flex flex-col gap-1.5">
        <Label>Strategy filter</Label>
        <Select value={scope.strategy || "__all__"} onValueChange={(v) => set("strategy", v === "__all__" || !v ? "" : v)}>
          <SelectTrigger>
            <SelectValue>{(v: string | null) => (!v || v === "__all__" ? "All strategies" : v)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All strategies</SelectItem>
            {strategies?.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Separator />

      {scope.agentic ? (
        // Agentic RAG's model is NOT a per-request choice like Simple RAG's chat
        // model above -- it's one shared, cached LLM instance for the whole app
        // (edenview_RAG.agentic_rag.config.get_shared_llm(), @lru_cache(maxsize=1)),
        // and changing it always requires an API server restart (see
        // api/routers/config.py's RESTART_REQUIRED_KEYS). Showing a picker here that
        // silently did nothing per-request would be misleading -- this is an
        // informational display + a link to where it's actually changed.
        <div className="flex flex-col gap-1.5">
          <Label>Agentic RAG model</Label>
          {modelSettings ? (
            <div className="flex flex-col gap-1 rounded-lg border border-dashed border-input px-2.5 py-2 text-xs text-muted-foreground">
              <span>
                Agent: <span className="text-foreground">{modelSettings.agent_model}</span>
              </span>
              <span>
                Vision: <span className="text-foreground">{modelSettings.agent_vision_model ?? "inherits agent model, if supported"}</span>
              </span>
              <span>
                Max iterations: <span className="text-foreground">{modelSettings.agent_max_iterations}</span>
              </span>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Loading…</p>
          )}
          <Link href="/settings" className="text-xs text-primary hover:underline">
            Change in Settings (needs a restart) →
          </Link>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          <Label>Chat model</Label>
          <Select
            value={scope.chatModel || "__default__"}
            onValueChange={(v) => set("chatModel", v === "__default__" || !v ? "" : v)}
          >
            <SelectTrigger>
              <SelectValue>
                {(v: string | null) =>
                  !v || v === "__default__" ? `Default${modelSettings ? ` (${modelSettings.chat_llm})` : ""}` : v
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__default__">
                Default {modelSettings ? `(${modelSettings.chat_llm})` : ""}
              </SelectItem>
              {system?.ollama.models.map((m) => (
                <SelectItem key={m.name} value={m.name}>
                  {m.name} · {m.size_gb} GB
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </aside>
  );
}
