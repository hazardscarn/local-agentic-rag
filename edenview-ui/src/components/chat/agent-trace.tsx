"use client";

import { useState } from "react";
import { Check, ChevronDown, ChevronRight, Loader2, X } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type { NodeRun, PipelineState, SubquestionThread } from "@/lib/pipeline-state";

// Human-readable label per node -- mirrors edenview_RAG/agentic_rag/prompts.py's
// AGENT_STATUS_LABELS/STATUS_LABELS so this reads the same as the single-line status
// text elsewhere in the chat UI.
const NODE_LABELS: Record<string, string> = {
  decompose: "Understanding your question",
  reworder: "Trying a different search phrasing",
  search_executor: "Searching your documents",
  vector_search: "Vector search",
  eval: "Checking whether the answer is complete",
  deep_search: "Looking for more information",
  get_pages_detailed: "Reading the full page",
  get_images: "Looking for related images",
  grep: "Searching for an exact term",
  get_answer_from_images: "Looking closer at a retrieved image",
  get_answer_from_detailed_pages: "Reading through the full page",
  answer_formatter: "Writing the answer",
};

function label(node: string): string {
  return NODE_LABELS[node] ?? node;
}

function StatusIcon({ status }: { status: NodeRun["status"] | SubquestionThread["status"] }) {
  if (status === "done") return <Check className="size-3.5 text-primary" />;
  if (status === "error") return <X className="size-3.5 text-destructive" />;
  if (status === "running") return <Loader2 className="size-3.5 animate-spin text-primary" />;
  return <div className="size-3.5 rounded-full border border-border" />;
}

function NodeRow({ run, depth = 0 }: { run: NodeRun; depth?: number }) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = run.children.length > 0;
  return (
    <div className={depth > 0 ? "ml-5 border-l border-border pl-3" : undefined}>
      <div className="flex items-center gap-2 py-1 text-sm">
        {hasChildren ? (
          <button type="button" onClick={() => setExpanded((e) => !e)} className="text-muted-foreground hover:text-foreground">
            {expanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          </button>
        ) : (
          <span className="size-3.5" />
        )}
        <StatusIcon status={run.status} />
        <span className="flex-1 truncate">{label(run.node)}</span>
        {run.status === "done" && run.duration_s != null && (
          <span className="shrink-0 text-xs text-muted-foreground">{run.duration_s.toFixed(1)}s</span>
        )}
      </div>
      {hasChildren && expanded && (
        <div className="flex flex-col">
          {run.children.map((child, i) => (
            <NodeRow key={`${child.node}-${i}`} run={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function ThreadCard({ thread }: { thread: SubquestionThread }) {
  const title = thread.total === 1 ? thread.text : `(${thread.index}/${thread.total}) ${thread.text}`;
  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex items-start gap-2">
          <div className="mt-0.5 shrink-0">
            <StatusIcon status={thread.status} />
          </div>
          <p className="text-xs leading-snug font-medium">{title}</p>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col">
        {thread.nodes.length === 0 ? (
          <p className="text-xs text-muted-foreground">Starting…</p>
        ) : (
          thread.nodes.map((run, i) => <NodeRow key={`${run.node}-${i}`} run={run} />)
        )}
      </CardContent>
    </Card>
  );
}

// The full step-by-step trace (decompose -> per-sub-question node timelines,
// including nested tool calls -> answer_formatter) -- deliberately just the content,
// no <aside>/collapse chrome, so it can be embedded either in a live side panel or,
// as of this component's introduction, inside a completed chat message's own
// collapsible "reasoning trace" disclosure (see chat-message.tsx).
export function AgentTrace({ state }: { state: PipelineState }) {
  const isEmpty = !state.decompose && state.threads.length === 0 && !state.answerFormatter;
  if (isEmpty) {
    return <p className="text-xs text-muted-foreground">{state.rootMessage ?? "Waiting to start…"}</p>;
  }
  return (
    <div className="flex flex-col gap-3">
      {state.decompose && (
        <div className="flex items-center gap-2 text-sm">
          <StatusIcon status={state.decompose.status} />
          <span className="flex-1 truncate">{label("decompose")}</span>
          {state.decompose.status === "done" && state.decompose.duration_s != null && (
            <span className="text-xs text-muted-foreground">{state.decompose.duration_s.toFixed(1)}s</span>
          )}
        </div>
      )}

      <div className="flex flex-col gap-2">
        {state.threads.map((thread) => (
          <ThreadCard key={thread.index} thread={thread} />
        ))}
      </div>

      {state.answerFormatter && (
        <div className="flex items-center gap-2 text-sm">
          <StatusIcon status={state.answerFormatter.status} />
          <span className="flex-1 truncate">{label("answer_formatter")}</span>
          {state.answerFormatter.status === "done" && state.answerFormatter.duration_s != null && (
            <span className="text-xs text-muted-foreground">{state.answerFormatter.duration_s.toFixed(1)}s</span>
          )}
        </div>
      )}
    </div>
  );
}
