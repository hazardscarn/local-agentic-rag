"use client";

import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PipelineState, NodeStatus, NodeRun } from "@/lib/pipeline-state";

// =====================================================================
// Unified Researcher Flow (NEW architecture) — LangGraph Studio-style graph.
// The graph's STRUCTURE is fixed and always fully drawn (Root → Researcher →
// Tools → Generating Answer), exactly like opening a graph in LangGraph Studio
// before a run starts. Only each node/edge's *status* (pending / running /
// done) changes live as events stream in — nothing ever appears, disappears,
// or reflows, so the shape stays stable and readable turn to turn.
// =====================================================================

/** The researcher's full, fixed tool roster (mirrors agent.py's researcher_tools),
 *  all grouped under one "Tools" node -- vector_search first (primary retrieval),
 *  the rest are selective deep-dive tools the researcher reaches for only on some
 *  turns. Each stays pending until actually invoked, same as an unvisited node in
 *  a LangGraph Studio graph. Only the call type/count is shown, never the live
 *  query/content text -- this is a structural graph view, not a search log. */
const TOOL_DEFS: Array<{ id: string; label: string }> = [
  { id: "vector_search", label: "Vector Search" },
  { id: "get_pages_detailed", label: "Get Page" },
  { id: "grep", label: "Grep" },
  { id: "get_images", label: "Get Images" },
  { id: "get_answer_from_images", label: "Image Q&A" },
  { id: "get_answer_from_detailed_pages", label: "Page Q&A" },
];

interface ToolNodeState {
  id: string;
  label: string;
  status: NodeStatus;
  calls: number;
}

/** Merge the researcher's live child tool-calls onto the fixed tool roster --
 *  a tool called more than once keeps its most recent (running-preferring) status. */
function toolNodeState(def: { id: string; label: string }, children: NodeRun[]): ToolNodeState {
  const calls = children.filter((c) => c.node === def.id);
  if (calls.length === 0) {
    return { id: def.id, label: def.label, status: "pending", calls: 0 };
  }
  const current = calls.find((c) => c.status === "running") ?? calls[calls.length - 1];
  return { id: def.id, label: def.label, status: current.status, calls: calls.length };
}

// ─── Layout constants (fixed geometry -- the graph shape never changes shape,
// only node/edge colors, so no layout math needs to react to live data). ───
const NODE_W = 172; // Researcher / Generating Answer box width
const ROOT_W = 86;
const ROOT_H = 24;
const RESEARCHER_H = 46;
const CHIP_COLS = 2;
const CHIP_W = 104;
const CHIP_H = 28;
const CHIP_GAP_X = 8;
const CHIP_GAP_Y = 8;
const BOX_PAD = 12;
const BOX_HEADER_H = 22;
const ANSWER_W = 172;
const ANSWER_H = 36;
const PAD_X = 16;
const GAP_ROOT_RESEARCHER = 28;
const GAP_RESEARCHER_BOX = 30;
const GAP_BOX_ANSWER = 30;

function agentBoxClass(status: NodeStatus): string {
  if (status === "running") return "fill-card stroke-primary";
  if (status === "done") return "fill-muted/40 stroke-primary/40";
  return "fill-muted/10 stroke-border/50";
}

function toolChipClass(status: NodeStatus): string {
  if (status === "running") return "fill-primary/15 stroke-primary";
  if (status === "done") return "fill-muted/30 stroke-muted-foreground/40";
  return "fill-transparent stroke-border/30";
}

function edgeClass(active: boolean): string {
  return active ? "stroke-primary/70" : "stroke-border/40";
}

function StatusIcon({ status, size = "size-3" }: { status: NodeStatus; size?: string }) {
  if (status === "running") return <Loader2 className={cn(size, "shrink-0 animate-spin text-primary")} />;
  if (status === "done") return <Check className={cn(size, "shrink-0 text-primary")} />;
  return <div className={cn(size, "shrink-0 rounded-full border border-border/50")} />;
}

function labelClass(status: NodeStatus): string {
  if (status === "running") return "text-primary";
  if (status === "done") return "text-muted-foreground/70";
  return "text-muted-foreground/40";
}

function ResearcherFlow({ state }: { state: PipelineState }) {
  if (!state.researcher && !state.rootAgent) return null;

  // Root node reflects root_agent (the AgentTool caller); once the researcher
  // itself has started, Root is necessarily behind us, so treat it as done even
  // if a "root_agent end" event hasn't landed yet (it fires after the answer
  // passes back through root, well after this graph stops caring about it).
  const rootStatus: NodeStatus = state.researcher ? "done" : state.rootAgent?.status ?? "pending";
  const researcherStatus: NodeStatus = state.researcher?.status ?? "pending";
  const researcherRunning = researcherStatus === "running";
  const hasAnswer = researcherStatus === "done";
  const children = state.researcher?.children ?? [];

  const tools = TOOL_DEFS.map((def) => toolNodeState(def, children));
  const anyToolActive = tools.some((t) => t.status !== "pending");
  const anyToolRunning = tools.some((t) => t.status === "running");
  const allFiredToolsDone = children.length > 0 && children.every((c) => c.status === "done");

  // No direct signal distinguishes "still deciding whether to search more" from
  // "now writing the final answer" -- both are just researcherStatus:"running".
  // Once every tool call that HAS fired has finished, the researcher is almost
  // certainly composing its answer rather than about to fire another (searches
  // all happen up front, deep-dives are capped at 2 rounds) -- close enough to
  // treat as "running" for a live, honest-feeling status without a backend change.
  const generatingStatus: NodeStatus = hasAnswer ? "done" : researcherRunning && allFiredToolsDone ? "running" : "pending";

  // ─── Fixed geometry ───
  const chipGridW = CHIP_COLS * CHIP_W + (CHIP_COLS - 1) * CHIP_GAP_X;
  const chipRows = Math.ceil(TOOL_DEFS.length / CHIP_COLS);
  const chipGridH = chipRows * CHIP_H + (chipRows - 1) * CHIP_GAP_Y;
  const boxW = chipGridW + BOX_PAD * 2;
  const boxH = BOX_HEADER_H + chipGridH + BOX_PAD * 2;

  const canvasW = boxW + PAD_X * 2;
  const centerX = canvasW / 2;

  const rootY = 10;
  const researcherY = rootY + ROOT_H + GAP_ROOT_RESEARCHER;
  const boxY = researcherY + RESEARCHER_H + GAP_RESEARCHER_BOX;
  const answerY = boxY + boxH + GAP_BOX_ANSWER;
  const canvasH = answerY + ANSWER_H + 14;

  const rootX = centerX - ROOT_W / 2;
  const researcherX = centerX - NODE_W / 2;
  const boxX = centerX - boxW / 2;
  const chipGridX = boxX + BOX_PAD;
  const chipGridY = boxY + BOX_HEADER_H + BOX_PAD;
  const answerX = centerX - ANSWER_W / 2;

  const chipPos = (i: number) => {
    const col = i % CHIP_COLS;
    const row = Math.floor(i / CHIP_COLS);
    return { x: chipGridX + col * (CHIP_W + CHIP_GAP_X), y: chipGridY + row * (CHIP_H + CHIP_GAP_Y) };
  };

  // Loop-back edge (Researcher <-> Tools, mirroring a real agent/tool cycle):
  // routed through the natural gutter to the right of the node column, since
  // the tools box is wider than the agent boxes above and below it. Anchored at
  // the box's TOP-right corner (level with the "Tools" header, not any individual
  // chip row) so it reads as "the box as a whole loops back" -- anchoring it at
  // the box's vertical center previously landed the curve right next to whichever
  // chip happened to sit there, reading as if that ONE tool specifically was being
  // called even when it was still pending.
  const loopX = boxX + boxW + 14;
  const loopStartY = boxY + BOX_HEADER_H / 2;
  const loopActive = researcherRunning && anyToolActive;

  const boxBorderClass = anyToolRunning ? "stroke-primary/60" : hasAnswer || allFiredToolsDone ? "stroke-primary/25" : "stroke-border/40";

  return (
    <svg viewBox={`0 0 ${canvasW} ${canvasH}`} width="100%" className="overflow-visible">
      <defs>
        <marker id="rf-arrow" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" className="fill-border" />
        </marker>
        <marker id="rf-arrow-active" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" className="fill-primary/70" />
        </marker>
        <filter id="rf-glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feFlood floodColor="var(--primary)" floodOpacity="0.35" />
          <feComposite in2="blur" operator="in" />
          <feMerge>
            <feMergeNode />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* ─── Edge: Root → Researcher ─── */}
      <line
        x1={centerX} y1={rootY + ROOT_H} x2={centerX} y2={researcherY}
        strokeWidth={1.4} markerEnd={researcherStatus !== "pending" ? "url(#rf-arrow-active)" : "url(#rf-arrow)"}
        className={edgeClass(researcherStatus !== "pending")}
      />

      {/* ─── Edge: Researcher → Tools box ─── */}
      <line
        x1={centerX} y1={researcherY + RESEARCHER_H} x2={centerX} y2={boxY}
        strokeWidth={1.4} markerEnd={anyToolActive ? "url(#rf-arrow-active)" : "url(#rf-arrow)"}
        className={edgeClass(anyToolActive)}
        strokeDasharray={anyToolActive ? undefined : "3 4"}
      />

      {/* ─── Loop-back edge: Tools → Researcher (the agent/tool cycle) ─── */}
      <path
        d={`M${loopX},${loopStartY} C${loopX + 16},${(boxY + researcherY) / 2} ${loopX + 10},${researcherY + RESEARCHER_H * 0.7} ${researcherX + NODE_W},${researcherY + RESEARCHER_H * 0.72}`}
        fill="none" strokeWidth={1.2}
        className={edgeClass(loopActive)}
        strokeDasharray="2 4"
        markerEnd={loopActive ? "url(#rf-arrow-active)" : "url(#rf-arrow)"}
      />

      {/* ─── Edge: Tools box → Generating Answer ─── */}
      <line
        x1={centerX} y1={boxY + boxH} x2={centerX} y2={answerY}
        strokeWidth={1.4} markerEnd={generatingStatus !== "pending" ? "url(#rf-arrow-active)" : "url(#rf-arrow)"}
        className={edgeClass(generatingStatus !== "pending")}
        strokeDasharray={generatingStatus === "pending" ? "3 4" : undefined}
      />

      {/* ─── Root node ─── */}
      <g>
        {rootStatus === "running" && <rect x={rootX - 3} y={rootY - 3} width={ROOT_W + 6} height={ROOT_H + 6} rx={14} className="fill-primary/10" filter="url(#rf-glow)" />}
        <rect x={rootX} y={rootY} width={ROOT_W} height={ROOT_H} rx={12} className={agentBoxClass(rootStatus)} strokeWidth={rootStatus === "running" ? 1.6 : 1.2} />
        <foreignObject x={rootX} y={rootY} width={ROOT_W} height={ROOT_H}>
          <div className="flex h-full w-full items-center justify-center gap-1.5">
            <StatusIcon status={rootStatus} size="size-2.5" />
            <span className={cn("text-[10px] font-medium", labelClass(rootStatus))}>Root</span>
          </div>
        </foreignObject>
      </g>

      {/* ─── Researcher node ─── */}
      <g>
        {researcherRunning && <rect x={researcherX - 4} y={researcherY - 4} width={NODE_W + 8} height={RESEARCHER_H + 8} rx={13} className="fill-primary/10" filter="url(#rf-glow)" />}
        <rect x={researcherX} y={researcherY} width={NODE_W} height={RESEARCHER_H} rx={10} className={agentBoxClass(researcherStatus)} strokeWidth={researcherRunning ? 2 : 1.5} />
        <foreignObject x={researcherX} y={researcherY} width={NODE_W} height={RESEARCHER_H}>
          <div className="flex h-full w-full items-center justify-center gap-2">
            <StatusIcon status={researcherStatus} size="size-4" />
            <span className={cn("text-xs font-semibold", labelClass(researcherStatus))}>Researcher</span>
          </div>
        </foreignObject>
      </g>

      {/* ─── Tools box: every tool the researcher can call, grouped together ─── */}
      <g>
        <rect x={boxX} y={boxY} width={boxW} height={boxH} rx={12} className={cn("fill-card/40", boxBorderClass)} strokeWidth={1.4} strokeDasharray={anyToolActive ? undefined : "4 4"} />
        <foreignObject x={boxX} y={boxY} width={boxW} height={BOX_HEADER_H}>
          <div className="flex h-full w-full items-center px-3">
            <span className="text-[9.5px] font-semibold uppercase tracking-wide text-muted-foreground/50">Tools</span>
          </div>
        </foreignObject>
        {tools.map((t, i) => {
          const { x, y } = chipPos(i);
          return (
            <g key={t.id}>
              <rect x={x} y={y} width={CHIP_W} height={CHIP_H} rx={13} className={toolChipClass(t.status)} strokeWidth={t.status === "running" ? 1.5 : 1} />
              <foreignObject x={x} y={y} width={CHIP_W} height={CHIP_H}>
                <div className="flex h-full w-full items-center justify-center gap-1 px-1.5">
                  <StatusIcon status={t.status} size="size-2.5" />
                  <span className={cn("truncate text-[9.5px] font-medium leading-none", labelClass(t.status))}>
                    {t.label}{t.calls > 1 ? ` ×${t.calls}` : ""}
                  </span>
                </div>
              </foreignObject>
            </g>
          );
        })}
      </g>

      {/* ─── Generating Answer node ─── */}
      <g>
        {generatingStatus === "running" && <rect x={answerX - 4} y={answerY - 4} width={ANSWER_W + 8} height={ANSWER_H + 8} rx={13} className="fill-primary/10" filter="url(#rf-glow)" />}
        <rect x={answerX} y={answerY} width={ANSWER_W} height={ANSWER_H} rx={10} className={agentBoxClass(generatingStatus)} strokeWidth={generatingStatus === "running" ? 2 : 1.5} />
        <foreignObject x={answerX} y={answerY} width={ANSWER_W} height={ANSWER_H}>
          <div className="flex h-full w-full items-center justify-center gap-2">
            <StatusIcon status={generatingStatus} size="size-3.5" />
            <span className={cn("text-xs font-semibold", labelClass(generatingStatus))}>Generating Answer</span>
          </div>
        </foreignObject>
      </g>
    </svg>
  );
}

// =====================================================================
// Legacy Multi-Agent Flowchart (original decompose→loop→answer)
// Kept as-is for backward compatibility with any existing consumers.
// =====================================================================

type StageKey = "decompose" | "reworder" | "search_executor" | "eval" | "deep_search" | "answer_formatter";
type StageStateLegacy = "pending" | "active" | "done";

const STAGE_LABELS_LEGACY: Record<StageKey, string> = {
  decompose: "Decompose",
  reworder: "Rephrase",
  search_executor: "Vector Search",
  eval: "Evaluate",
  deep_search: "Deep Search",
  answer_formatter: "Answer",
};

const DEEP_SEARCH_TOOLS_LEGACY = [
  { node: "get_pages_detailed", label: "Page" },
  { node: "get_images", label: "Images" },
  { node: "grep", label: "Grep" },
  { node: "get_answer_from_images", label: "Img Q&A" },
  { node: "get_answer_from_detailed_pages", label: "Page Q&A" },
] as const;

function stageStateLegacy(state: PipelineState, key: StageKey): StageStateLegacy {
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
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return "pending";
  const runs = thread.nodes.filter((n) => n.node === key);
  if (runs.length === 0) return "pending";
  if (runs.some((n) => n.status === "running")) return "active";
  return "done";
}

function activeSearchToolLegacy(state: PipelineState): string | null {
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return null;
  const runs = thread.nodes.filter((n) => n.node === "search_executor");
  if (runs.length === 0 || runs[runs.length - 1].children.length === 0) return null;
  return "vector search";
}

function deepSearchToolStateLegacy(state: PipelineState, toolNode: string): StageStateLegacy {
  const thread = state.threads[state.threads.length - 1];
  if (!thread) return "pending";
  const children = thread.nodes.filter((n) => n.node === "deep_search").flatMap((r) => r.children).filter((c) => c.node === toolNode);
  if (children.length === 0) return "pending";
  if (children.some((c) => c.status === "running")) return "active";
  return "done";
}

function BoxLegacy({ x, y, w, h, keyName, state: st, subtitle }: { x: number; y: number; w: number; h: number; keyName: StageKey; state: StageStateLegacy; subtitle?: string | null }) {
  return (
    <g>
      {st === "active" && (
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
          st === "active" && "fill-card stroke-primary",
          st === "done" && "fill-muted stroke-primary/40",
          st === "pending" && "fill-transparent stroke-border",
        )}
        strokeWidth={st === "active" ? 2 : 1.5}
        strokeDasharray={st === "pending" ? "4 3" : undefined}
      />
      <foreignObject x={x} y={y} width={w} height={h}>
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-1 text-center">
          <div className="flex items-center gap-1.5">
            {st === "active" && <Loader2 className="size-4 shrink-0 animate-spin text-primary" />}
            {st === "done" && <Check className="size-4 shrink-0 text-primary" />}
            <span className={cn("truncate text-xs font-medium", st === "pending" && "text-muted-foreground")}>
              {STAGE_LABELS_LEGACY[keyName]}
            </span>
          </div>
          {subtitle && <span className="truncate text-[10px] leading-none text-muted-foreground">{subtitle}</span>}
        </div>
      </foreignObject>
    </g>
  );
}

function ToolChipLegacy({ label, state: st }: { label: string; state: StageStateLegacy }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 leading-[16px] text-[9.5px]",
        st === "active" && "border-primary text-primary",
        st === "done" && "border-primary/40 bg-background text-foreground",
        st === "pending" && "border-border/50 text-muted-foreground/40",
      )}
    >
      {st === "active" && <Loader2 className="size-2.5 shrink-0 animate-spin" />}
      {st === "done" && <Check className="size-2.5 shrink-0" />}
      {label}
    </span>
  );
}

function DeepSearchBoxLegacy({ x, y, w, h: hBox, state: st, pipelineState }: { x: number; y: number; w: number; h: number; state: StageStateLegacy; pipelineState: PipelineState }) {
  return (
    <g>
      {st === "active" && (
        <rect x={x} y={y} width={w} height={hBox} rx={10} className="fill-primary/25" filter="url(#eaf-glow)" />
      )}
      <rect
        x={x}
        y={y}
        width={w}
        height={hBox}
        rx={8}
        className={cn(
          "transition-colors",
          st === "active" && "fill-card stroke-primary",
          st === "done" && "fill-muted stroke-primary/40",
          st === "pending" && "fill-transparent stroke-border",
        )}
        strokeWidth={st === "active" ? 2 : 1.5}
        strokeDasharray={st === "pending" ? "4 3" : undefined}
      />
      <foreignObject x={x} y={y} width={w} height={hBox}>
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 px-3 py-2 text-center">
          <div className="flex items-center gap-1.5">
            {st === "active" && <Loader2 className="size-4 shrink-0 animate-spin text-primary" />}
            {st === "done" && <Check className="size-4 shrink-0 text-primary" />}
            <span className={cn("truncate text-xs font-medium", st === "pending" && "text-muted-foreground")}>
              Deep Search
            </span>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-1.5">
            {DEEP_SEARCH_TOOLS_LEGACY.map((t) => (
              <ToolChipLegacy key={t.node} label={t.label} state={deepSearchToolStateLegacy(pipelineState, t.node)} />
            ))}
          </div>
        </div>
      </foreignObject>
    </g>
  );
}

// Fixed layout constants for the legacy diagram.
const W_LEGACY = 320;
const BOX_W_LEGACY = 140;
const BOX_H_LEGACY = 46;

function LegacyFlowchart({ state }: { state: PipelineState }) {
  const decomposeX = (W_LEGACY - BOX_W_LEGACY) / 2;
  const decomposeY = 20;
  const rephraseX = (W_LEGACY - BOX_W_LEGACY) / 2;
  const rephraseY = 116;
  const searchX = 8;
  const searchY = 214;
  const evalX = W_LEGACY - BOX_W_LEGACY - 8;
  const evalY = 214;
  const deepSearchW = 284;
  const deepSearchX = (W_LEGACY - deepSearchW) / 2;
  const deepSearchY = 336;
  const deepSearchH = 96;
  const answerX = (W_LEGACY - BOX_W_LEGACY) / 2;
  const answerY = 480;
  const H = answerY + BOX_H_LEGACY + 20;

  const edgeClass = "stroke-border";
  const arrowMarker = "url(#eaf-arrow)";

  return (
    <svg viewBox={`0 0 ${W_LEGACY} ${H}`} width="100%" className="overflow-visible">
      <defs>
        <marker id="eaf-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" className="fill-border" />
        </marker>
        <filter id="eaf-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {/* Loop grouping */}
      <rect
        x={4}
        y={rephraseY - 22}
        width={W_LEGACY - 8}
        height={evalY + BOX_H_LEGACY + 20 - (rephraseY - 22)}
        rx={12}
        className="fill-muted/20 stroke-border"
        strokeDasharray="3 4"
        strokeWidth={1}
      />
      <text x={14} y={rephraseY - 28} className="fill-muted-foreground text-[9px] uppercase tracking-wide">
        research loop (per sub-question)
      </text>

      {/* Decompose -> Vector Search */}
      <path
        d={`M${decomposeX + 12},${decomposeY + BOX_H_LEGACY} C${decomposeX - 24},${decomposeY + BOX_H_LEGACY + 36} ${searchX + 12},${searchY - 36} ${searchX + 34},${searchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Triangular loop */}
      <path
        d={`M${rephraseX + 22},${rephraseY + BOX_H_LEGACY} C${rephraseX},${rephraseY + BOX_H_LEGACY + 30} ${searchX + 44},${searchY - 24} ${searchX + 60},${searchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />
      <line x1={searchX + BOX_W_LEGACY} y1={searchY + BOX_H_LEGACY / 2} x2={evalX} y2={evalY + BOX_H_LEGACY / 2} className={edgeClass} strokeWidth={1.5} markerEnd={arrowMarker} />
      <path
        d={`M${evalX + 22},${evalY} C${evalX},${evalY - 30} ${rephraseX + BOX_W_LEGACY - 12},${rephraseY + BOX_H_LEGACY + 30} ${rephraseX + BOX_W_LEGACY - 28},${rephraseY + BOX_H_LEGACY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Evaluate <-> Deep Search */}
      <path
        d={`M${evalX + 40},${evalY + BOX_H_LEGACY} C${evalX},${evalY + BOX_H_LEGACY + 40} ${deepSearchX + deepSearchW - 60},${deepSearchY - 30} ${deepSearchX + deepSearchW - 40},${deepSearchY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />
      <path
        d={`M${deepSearchX + deepSearchW - 20},${deepSearchY} C${deepSearchX + deepSearchW + 20},${deepSearchY - 40} ${evalX + 70},${evalY + BOX_H_LEGACY + 50} ${evalX + 70},${evalY + BOX_H_LEGACY}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      {/* Evaluate -> Answer */}
      <path
        d={`M${evalX + 10},${evalY + BOX_H_LEGACY} C${evalX - 40},${evalY + 120} ${answerX + BOX_W_LEGACY + 30},${answerY - 60} ${answerX + BOX_W_LEGACY - 6},${answerY + 10}`}
        fill="none"
        className={edgeClass}
        strokeWidth={1.5}
        markerEnd={arrowMarker}
      />

      <line x1={deepSearchX + deepSearchW / 2} y1={deepSearchY + deepSearchH} x2={answerX + BOX_W_LEGACY / 2} y2={answerY} className={edgeClass} strokeWidth={1.5} strokeDasharray="2 4" />

      <BoxLegacy x={decomposeX} y={decomposeY} w={BOX_W_LEGACY} h={BOX_H_LEGACY} keyName="decompose" state={stageStateLegacy(state, "decompose")} />
      <BoxLegacy x={rephraseX} y={rephraseY} w={BOX_W_LEGACY} h={BOX_H_LEGACY} keyName="reworder" state={stageStateLegacy(state, "reworder")} />
      <BoxLegacy x={searchX} y={searchY} w={BOX_W_LEGACY} h={BOX_H_LEGACY} keyName="search_executor" state={stageStateLegacy(state, "search_executor")} subtitle={activeSearchToolLegacy(state)} />
      <BoxLegacy x={evalX} y={evalY} w={BOX_W_LEGACY} h={BOX_H_LEGACY} keyName="eval" state={stageStateLegacy(state, "eval")} />
      <DeepSearchBoxLegacy x={deepSearchX} y={deepSearchY} w={deepSearchW} h={deepSearchH} state={stageStateLegacy(state, "deep_search")} pipelineState={state} />
      <BoxLegacy x={answerX} y={answerY} w={BOX_W_LEGACY} h={BOX_H_LEGACY} keyName="answer_formatter" state={stageStateLegacy(state, "answer_formatter")} />
    </svg>
  );
}

// =====================================================================
// Main export -- selects the right flowchart based on architecture
// =====================================================================

/** Returns true when we should render the unified researcher flow. */
function isResearcherMode(state: PipelineState): boolean {
  return state.researcher !== null;
}

export function AgentFlowchart({ state }: { state: PipelineState }) {
  if (isResearcherMode(state)) {
    return <ResearcherFlow state={state} />;
  }
  return <LegacyFlowchart state={state} />;
}

// Re-export stageStateLegacy for any consumers that still need to query legacy phases.
export { stageStateLegacy as stageState };
