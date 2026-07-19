"use client";

import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PipelineState } from "@/lib/pipeline-state";

// A static diagram of the pipeline's fixed topology (it never changes shape --
// always decompose -> [rephrase? -> vector search -> evaluate -> deep_search?]* ->
// answer, see edenview_RAG/agentic_rag/agent.py) -- NOT a general-purpose graph
// renderer. Deliberately hand-laid-out with plain SVG rather than a graph library
// (react-flow/@xyflow etc.): those exist for graphs whose SHAPE is dynamic
// (user-editable, auto-layout, arbitrary node counts) and would be real, unneeded
// weight for a diagram that's always these same boxes. Rephrase/Vector Search/
// Evaluate are drawn as a triangular loop (they can repeat several times per
// sub-question), with Evaluate as the decision point branching out to Deep Search
// or Answer -- mirroring the real control flow in agent.py's SubquestionLoop.
// Boxes light up from the exact same PipelineState the trace view (agent-trace.tsx)
// already derives from -- no new state, just a different rendering of it.

type StageKey = "decompose" | "reworder" | "search_executor" | "eval" | "deep_search" | "answer_formatter";
type StageState = "pending" | "active" | "done";

const STAGE_LABELS: Record<StageKey, string> = {
  decompose: "Decompose",
  reworder: "Rephrase",
  search_executor: "Vector Search",
  eval: "Evaluate",
  deep_search: "Deep Search",
  answer_formatter: "Answer",
};

const DEEP_SEARCH_TOOLS = [
  { node: "get_pages_detailed", label: "Page" },
  { node: "get_images", label: "Images" },
  { node: "grep", label: "Grep" },
  { node: "get_answer_from_images", label: "Img Q&A" },
  { node: "get_answer_from_detailed_pages", label: "Page Q&A" },
] as const;

export function stageState(state: PipelineState, key: StageKey): StageState {
  if (key === "decompose") {
    if (state.decompose?.status === "running") return "active";
    if (state.decompose?.status === "done") return "done";
    return "pending";
  }
  if (key === "answer_formatter") {
    if (state.answerFormatter?.status === "running") return "active";
    if (state.answerFormatter?.status === "done") return "done";
    return "pending";
  }
  // Sub-questions run strictly sequentially -- the last thread is the only one that
  // can possibly be live right now.
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return "pending";
  const runs = thread.nodes.filter((n) => n.node === key);
  if (runs.length === 0) return "pending";
  if (runs.some((n) => n.status === "running")) return "active";
  return "done"; // ran at least once; may run again if the loop revisits it
}

// The most recent tool call fired by search_executor's last invocation in the
// currently-active thread (e.g. "vector search") -- search_executor only ever has
// one possible tool, so a single subtitle is enough (unlike Deep Search, which can
// invoke several different tools -- see DeepSearchToolChips below).
function activeSearchTool(state: PipelineState): string | null {
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return null;
  const runs = thread.nodes.filter((n) => n.node === "search_executor");
  if (runs.length === 0 || runs[runs.length - 1].children.length === 0) return null;
  return "vector search";
}

// Per-tool status across EVERY deep_search invocation in the active thread (it can
// run more than once per sub-question, and can call several of its 5 tools each
// time) -- "done" if that tool has fired at least once and isn't currently running,
// "active" if it's running right now, "pending" if never called.
function deepSearchToolState(state: PipelineState, toolNode: string): StageState {
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return "pending";
  const children = thread.nodes.filter((n) => n.node === "deep_search").flatMap((r) => r.children).filter((c) => c.node === toolNode);
  if (children.length === 0) return "pending";
  if (children.some((c) => c.status === "running")) return "active";
  return "done";
}

function Box({
  x,
  y,
  w,
  h,
  keyName,
  state,
  subtitle,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  keyName: StageKey;
  state: StageState;
  subtitle?: string | null;
}) {
  return (
    <g>
      {state === "active" && (
        <rect x={x} y={y} width={w} height={h} rx={10} className="fill-primary/25" filter="url(#eaf-glow)" />
      )}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        className={cn(
          "transition-colors",
          state === "active" && "fill-card stroke-primary",
          state === "done" && "fill-muted stroke-primary/40",
          state === "pending" && "fill-transparent stroke-border",
        )}
        strokeWidth={state === "active" ? 2 : 1.5}
        strokeDasharray={state === "pending" ? "4 3" : undefined}
      />
      <foreignObject x={x} y={y} width={w} height={h}>
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-1 text-center">
          <div className="flex items-center gap-1.5">
            {state === "active" && <Loader2 className="size-4 shrink-0 animate-spin text-primary" />}
            {state === "done" && <Check className="size-4 shrink-0 text-primary" />}
            <span className={cn("truncate text-xs font-medium", state === "pending" && "text-muted-foreground")}>
              {STAGE_LABELS[keyName]}
            </span>
          </div>
          {subtitle && <span className="truncate text-[10px] leading-none text-muted-foreground">{subtitle}</span>}
        </div>
      </foreignObject>
    </g>
  );
}

function ToolChip({ label, state }: { label: string; state: StageState }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 leading-[16px] text-[9.5px]",
        state === "active" && "border-primary text-primary",
        state === "done" && "border-primary/40 bg-background text-foreground",
        state === "pending" && "border-border/50 text-muted-foreground/40",
      )}
    >
      {state === "active" && <Loader2 className="size-2.5 shrink-0 animate-spin" />}
      {state === "done" && <Check className="size-2.5 shrink-0" />}
      {label}
    </span>
  );
}

function DeepSearchBox({
  x,
  y,
  w,
  h,
  state,
  pipelineState,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  state: StageState;
  pipelineState: PipelineState;
}) {
  return (
    <g>
      {state === "active" && (
        <rect x={x} y={y} width={w} height={h} rx={10} className="fill-primary/25" filter="url(#eaf-glow)" />
      )}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        className={cn(
          "transition-colors",
          state === "active" && "fill-card stroke-primary",
          state === "done" && "fill-muted stroke-primary/40",
          state === "pending" && "fill-transparent stroke-border",
        )}
        strokeWidth={state === "active" ? 2 : 1.5}
        strokeDasharray={state === "pending" ? "4 3" : undefined}
      />
      <foreignObject x={x} y={y} width={w} height={h}>
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 px-3 py-2 text-center">
          <div className="flex items-center gap-1.5">
            {state === "active" && <Loader2 className="size-4 shrink-0 animate-spin text-primary" />}
            {state === "done" && <Check className="size-4 shrink-0 text-primary" />}
            <span className={cn("truncate text-xs font-medium", state === "pending" && "text-muted-foreground")}>
              Deep Search
            </span>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-1.5">
            {DEEP_SEARCH_TOOLS.map((t) => (
              <ToolChip key={t.node} label={t.label} state={deepSearchToolState(pipelineState, t.node)} />
            ))}
          </div>
        </div>
      </foreignObject>
    </g>
  );
}

// Fixed layout constants -- this diagram's shape never changes, so plain hand-tuned
// coordinates (not a computed/measured layout) are fine. Sized generously (bigger
// boxes, more vertical breathing room between stages) rather than compactly -- the
// panel itself runs the full viewport height, and a small, tightly-packed diagram
// just left a large dead gap underneath it. Deep Search is now a wide, centered box
// (not a smaller one squeezed to Eval's side) so its 5 tool chips lay out in one
// clean row instead of wrapping awkwardly in a cramped space.
const W = 320;
const BOX_W = 140;
const BOX_H = 46;

export function AgentFlowchart({ state }: { state: PipelineState }) {
  const decomposeX = (W - BOX_W) / 2;
  const decomposeY = 20;
  const rephraseX = (W - BOX_W) / 2;
  const rephraseY = 116;
  const searchX = 8;
  const searchY = 214;
  const evalX = W - BOX_W - 8;
  const evalY = 214;
  const deepSearchW = 284;
  const deepSearchX = (W - deepSearchW) / 2;
  const deepSearchY = 336;
  const deepSearchH = 96;
  const answerX = (W - BOX_W) / 2;
  const answerY = 480;
  const H = answerY + BOX_H + 20;

  const edgeClass = "stroke-border";
  const arrowMarker = "url(#eaf-arrow)";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible">
      <defs>
        <marker id="eaf-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" className="fill-border" />
        </marker>
        <filter id="eaf-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {/* Loop grouping -- Rephrase/Vector Search/Evaluate form a triangular cycle
          that can repeat several times per sub-question; the dashed box makes that
          "this part can repeat" reading explicit rather than just implicit in the
          arrows. */}
      <rect
        x={4}
        y={rephraseY - 22}
        width={W - 8}
        height={evalY + BOX_H + 20 - (rephraseY - 22)}
        rx={12}
        className="fill-muted/20 stroke-border"
        strokeDasharray="3 4"
        strokeWidth={1}
      />
      <text x={14} y={rephraseY - 28} className="fill-muted-foreground text-[9px] uppercase tracking-wide">
        research loop (per sub-question)
      </text>

      {/* Decompose -> Vector Search (the real first action -- rephrase is skipped
          on a sub-question's first pass, seeded directly, see agent.py's
          SubquestionLoop) */}
      <path
        d={`M${decomposeX + 12},${decomposeY + BOX_H} C${decomposeX - 24},${decomposeY + BOX_H + 36} ${searchX + 12},${searchY - 36} ${searchX + 34},${searchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Triangular loop: Rephrase -> Vector Search -> Evaluate -> Rephrase */}
      <path
        d={`M${rephraseX + 22},${rephraseY + BOX_H} C${rephraseX},${rephraseY + BOX_H + 30} ${searchX + 44},${searchY - 24} ${searchX + 60},${searchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />
      <line
        x1={searchX + BOX_W}
        y1={searchY + BOX_H / 2}
        x2={evalX}
        y2={evalY + BOX_H / 2}
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />
      <path
        d={`M${evalX + 22},${evalY} C${evalX},${evalY - 30} ${rephraseX + BOX_W - 12},${rephraseY + BOX_H + 30} ${rephraseX + BOX_W - 28},${rephraseY + BOX_H}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Evaluate <-> Deep Search (branch out and back) -- Deep Search is now
          centered below the loop, so both lines fan in from/to Eval's bottom
          rather than sitting side by side. */}
      <path
        d={`M${evalX + 40},${evalY + BOX_H} C${evalX},${evalY + BOX_H + 40} ${deepSearchX + deepSearchW - 60},${deepSearchY - 30} ${deepSearchX + deepSearchW - 40},${deepSearchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />
      <path
        d={`M${deepSearchX + deepSearchW - 20},${deepSearchY} C${deepSearchX + deepSearchW + 20},${deepSearchY - 40} ${evalX + 70},${evalY + BOX_H + 50} ${evalX + 70},${evalY + BOX_H}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Evaluate -> Answer (exit once sufficient) */}
      <path
        d={`M${evalX + 10},${evalY + BOX_H} C${evalX - 40},${evalY + 120} ${answerX + BOX_W + 30},${answerY - 60} ${answerX + BOX_W - 6},${answerY + 10}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Deep Search -> Answer isn't a real edge (deep_search always loops back
          through Eval first) -- but visually Deep Search sits directly above
          Answer, so a short straight connector reads naturally without implying
          a shortcut exists. */}
      <line
        x1={deepSearchX + deepSearchW / 2}
        y1={deepSearchY + deepSearchH}
        x2={answerX + BOX_W / 2}
        y2={answerY}
        className={edgeClass}
        strokeWidth={1.5}
        strokeDasharray="2 4"
      />

      <Box x={decomposeX} y={decomposeY} w={BOX_W} h={BOX_H} keyName="decompose" state={stageState(state, "decompose")} />
      <Box x={rephraseX} y={rephraseY} w={BOX_W} h={BOX_H} keyName="reworder" state={stageState(state, "reworder")} />
      <Box
        x={searchX}
        y={searchY}
        w={BOX_W}
        h={BOX_H}
        keyName="search_executor"
        state={stageState(state, "search_executor")}
        subtitle={activeSearchTool(state)}
      />
      <Box x={evalX} y={evalY} w={BOX_W} h={BOX_H} keyName="eval" state={stageState(state, "eval")} />
      <DeepSearchBox
        x={deepSearchX}
        y={deepSearchY}
        w={deepSearchW}
        h={deepSearchH}
        state={stageState(state, "deep_search")}
        pipelineState={state}
      />
      <Box x={answerX} y={answerY} w={BOX_W} h={BOX_H} keyName="answer_formatter" state={stageState(state, "answer_formatter")} />
    </svg>
  );
}
