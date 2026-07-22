// Pure types + reducer (no React) for turning the raw AgenticStatusEvent stream
// (edenview_RAG/agentic_rag/callbacks.py's track_agent_start/end + track_tool_start/
// end, forwarded verbatim by runtime.py::run_turn_stream and api/routers/chat.py's
// chat_stream) into a structured view of the pipeline's live progress -- one entry
// per decomposed sub-question, each with its own independent reword/search/eval/
// deep-search timeline, since those node names repeat across sub-questions and loop
// iterations (a flat Record<node, status> would only ever show the LAST one).
//
// Mirrors the real pipeline shape (edenview_RAG/agentic_rag/agent.py):
// NEW: question_capture -> researcher (single unified agent with internal parallel tool calls).
// OLD (legacy, kept for backward compat):
//   question_capture -> decompose -> subquestion_orchestrator -> [subquestion_loop ->
//   [reworder?, search_executor(+vector_search), eval, deep_search?(+its tool calls)]] -> answer_formatter.

import type { AgenticStatusEvent } from "./types";

export type NodeStatus = "pending" | "running" | "done" | "error";

export interface NodeRun {
  node: string; // agent or tool name, e.g. "search_executor" | "vector_search"
  kind: "agent" | "tool";
  status: NodeStatus;
  message?: string;
  duration_s?: number | null;
  startedAt: number; // Date.now() at "start" -- lets a live view tick "Ns..." before duration_s arrives
  children: NodeRun[]; // tool calls fired during this agent's own turn (only ever non-empty for kind:"agent")
}

export interface SubquestionThread {
  index: number;
  total: number;
  text: string;
  status: NodeStatus; // "running" until answer_formatter starts (all research is finished by then, by construction)
  nodes: NodeRun[];
}

export interface PipelineState {
  // Unified researcher agent state (NEW architecture)
  researcher: NodeRun | null;
  // root_agent wraps researcher as an AgentTool call (see agent.py) -- tracked
  // separately so the flowchart can show a real "Start" node ahead of Researcher
  // instead of only appearing once the researcher's own first event arrives.
  rootAgent: NodeRun | null;
  // Legacy fields (kept for backward compat -- never written by new pipeline but still read)
  questionCapture: NodeRun | null;
  decompose: NodeRun | null;
  threads: SubquestionThread[];
  answerFormatter: NodeRun | null;
  rootMessage: string | null; // last node-less relayed message
}

export const initialPipelineState: PipelineState = {
  researcher: null,
  rootAgent: null,
  questionCapture: null,
  decompose: null,
  threads: [],
  answerFormatter: null,
  rootMessage: null,
};

// Tool names dispatched as leaves under whichever agent (search_executor/deep_search)
// invoked them, rather than as their own top-level row in a thread's node list --
// mirrors prompts.py's STATUS_LABELS keys exactly.
const TOOL_NODE_NAMES = new Set([
  "vector_search",
  "get_pages_detailed",
  "get_images",
  "grep",
  "get_answer_from_images",
  "get_answer_from_detailed_pages",
]);

// Top-level agent names rendered as their own timeline entry within a thread.
// subquestion_loop itself is deliberately excluded -- its start/end wraps the whole
// thread's own timeline and adds no information beyond what the thread's child
// nodes already show.
const THREAD_AGENT_NODE_NAMES = new Set(["reworder", "search_executor", "eval", "deep_search"]);

function startNodeRun(node: string, kind: "agent" | "tool", message?: string): NodeRun {
  return { node, kind, status: "running", message, startedAt: Date.now(), children: [] };
}

function endMostRecentRunning(nodes: NodeRun[], node: string, duration_s: number | null | undefined): NodeRun[] {
  const idx = [...nodes].reverse().findIndex((n) => n.node === node && n.status === "running");
  if (idx === -1) return nodes; // end with no matching start seen -- drop rather than fabricate a row
  const realIdx = nodes.length - 1 - idx;
  return nodes.map((n, i) => (i === realIdx ? { ...n, status: "done", duration_s } : n));
}

function upsertSingleton(current: NodeRun | null, event: AgenticStatusEvent, node: string): NodeRun | null {
  if (event.phase === "start") return startNodeRun(node, "agent", event.message);
  if (event.phase === "end") {
    return current ? { ...current, status: "done", duration_s: event.duration_s } : null;
  }
  return current;
}

export function pipelineReducer(state: PipelineState, event: AgenticStatusEvent): PipelineState {
  const { node, phase } = event;

  // root_agent wraps researcher as an AgentTool call -- gives the flowchart a
  // real "Start" node that lights up before researcher's own first event.
  if (node === "root_agent") {
    return { ...state, rootAgent: upsertSingleton(state.rootAgent, event, node) };
  }

  // Unified researcher agent (NEW architecture) -- single agent with parallel tool calls.
  if (node === "researcher") {
    if (phase === "start") {
      return { ...state, researcher: startNodeRun(node, "agent", event.message || "Researching...") };
    }
    if (phase === "end" && state.researcher) {
      return { ...state, researcher: { ...state.researcher, status: "done", duration_s: event.duration_s } };
    }
    return state;
  }

  // Node-less events relay a human-readable message directly (no agent/tool node).
  if (!node) {
    return event.message ? { ...state, rootMessage: event.message } : state;
  }

  // When researcher is active (running), attach tool events to the researcher's children.
  // This is how we capture vector_search calls, deep-dive tools, etc. under the single
  // researcher node — mirroring what callbacks.track_tool_start/end emit during the turn.
  if (state.researcher && state.researcher.status === "running") {
    const isTool = TOOL_NODE_NAMES.has(node);
    if (!isTool) return state; // unrecognized for researcher mode

    if (phase === "start") {
      const childRun = startNodeRun(node, "tool", event.message || node);
      return { ...state, researcher: { ...state.researcher, children: [...state.researcher.children, childRun] } };
    }
    if (phase === "end" && state.researcher) {
      // Find the matching running tool call under researcher and mark it done.
      const newChildren = [...state.researcher.children];
      let foundIdx = -1;
      for (let i = newChildren.length - 1; i >= 0; i--) {
        if (newChildren[i].node === node && newChildren[i].status === "running") {
          foundIdx = i;
          break;
        }
      }
      if (foundIdx === -1) return state; // no matching start seen -- drop
      newChildren[foundIdx] = { ...newChildren[foundIdx], status: "done", duration_s: event.duration_s };
      return { ...state, researcher: { ...state.researcher, children: newChildren } };
    }
  }

  // Legacy top-level agent names (kept for backward compatibility).
  if (node === "question_capture") {
    return { ...state, questionCapture: upsertSingleton(state.questionCapture, event, node) };
  }

  if (node === "decompose") {
    return { ...state, decompose: upsertSingleton(state.decompose, event, node) };
  }

  if (node === "subquestion_orchestrator") {
    if (phase === "start" && event.subquestion_index != null) {
      const thread: SubquestionThread = {
        index: event.subquestion_index,
        total: event.subquestion_total ?? 1,
        text: event.subquestion_text ?? event.message ?? "",
        status: "running",
        nodes: [],
      };
      // A whole-turn retry (see runtime.py's empty-answer retry) re-fires this
      // same sub-question's "start" from scratch without the frontend ever
      // resetting `pipeline` in between (that only happens in page.tsx's send(),
      // for a brand-new user-initiated turn) -- replace the stale thread for
      // this index instead of appending a duplicate, which previously produced
      // two threads sharing the same `index` and a React duplicate-key warning
      // on ThreadCard's `key={thread.index}`.
      const existingIdx = state.threads.findIndex((t) => t.index === thread.index);
      const threads =
        existingIdx === -1
          ? [...state.threads, thread]
          : state.threads.map((t, i) => (i === existingIdx ? thread : t));
      return { ...state, threads };
    }
    // The orchestrator's own overall start/end (no subquestion_index) wraps every
    // sub-question -- no separate rendering value beyond the per-thread cards.
    return state;
  }

  if (node === "answer_formatter") {
    const answerFormatter = upsertSingleton(state.answerFormatter, event, node);
    if (phase === "start") {
      // By construction (SequentialAgent order), every sub-question's own research
      // has finished before answer_formatter ever starts.
      return {
        ...state,
        answerFormatter,
        threads: state.threads.map((t) => (t.status === "running" ? { ...t, status: "done" } : t)),
      };
    }
    return { ...state, answerFormatter };
  }

  if (node === "subquestion_loop") {
    // Wraps one sub-question's whole timeline -- rendering the thread card itself
    // already conveys this, no separate row needed.
    return state;
  }

  // Everything below belongs to one specific sub-question's research thread.
  const threadIndex = state.threads.findIndex((t) => t.index === event.subquestion_index);
  const resolvedIndex = threadIndex !== -1 ? threadIndex : state.threads.length - 1;
  if (resolvedIndex < 0) return state; // no thread to attach to yet -- drop defensively

  const thread = state.threads[resolvedIndex];
  const isTool = TOOL_NODE_NAMES.has(node);
  const isThreadAgent = THREAD_AGENT_NODE_NAMES.has(node);
  if (!isTool && !isThreadAgent) return state; // unrecognized node name -- ignore rather than guess

  let nodes: NodeRun[];
  if (isTool) {
    // Attach as a child under the last still-running agent row (its actual
    // caller); if none is open (shouldn't normally happen), fall back to a
    // top-level row so the event isn't silently lost.
    const parentIdx = [...thread.nodes].reverse().findIndex((n) => n.kind === "agent" && n.status === "running");
    if (parentIdx === -1) {
      nodes =
        phase === "start"
          ? [...thread.nodes, startNodeRun(node, "tool", event.message)]
          : endMostRecentRunning(thread.nodes, node, event.duration_s);
    } else {
      const realParentIdx = thread.nodes.length - 1 - parentIdx;
      nodes = thread.nodes.map((n, i) => {
        if (i !== realParentIdx) return n;
        const children =
          phase === "start"
            ? [...n.children, startNodeRun(node, "tool", event.message)]
            : endMostRecentRunning(n.children, node, event.duration_s);
        return { ...n, children };
      });
    }
  } else {
    // Same agent fires once per loop iteration -- never overwrite on "start".
    nodes =
      phase === "start"
        ? [...thread.nodes, startNodeRun(node, "agent", event.message)]
        : endMostRecentRunning(thread.nodes, node, event.duration_s);
  }

  const threads = state.threads.map((t, i) => (i === resolvedIndex ? { ...t, nodes } : t));
  return { ...state, threads };
}
