"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getChatSession, reattachChatStream, runChat, runChatStream } from "@/lib/api";
import { ChatScopePanel, type ChatScope } from "@/components/chat/chat-scope-panel";
import { ChatMessage, type ChatTurn } from "@/components/chat/chat-message";
import { ChatSessionList } from "@/components/chat/chat-session-list";
import { GroundingPanel, type GroundingTarget } from "@/components/chat/grounding-panel";
import { AgentPipelinePanel } from "@/components/chat/agent-pipeline-panel";
import { AgentTrace } from "@/components/chat/agent-trace";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import type { AgenticStatusEvent, RetrievalHit } from "@/lib/types";
import { pipelineReducer, initialPipelineState, type PipelineState } from "@/lib/pipeline-state";
import { cn } from "@/lib/utils";
import { Send, Loader2, Sparkles, PanelRightOpen, PanelLeftOpen, ChevronRight } from "lucide-react";

const DEFAULT_SCOPE: ChatScope = {
  collectionNames: [],
  topK: 5,
  useReranker: true,
  strategy: "",
  chatModel: "",
  agentic: false,
};

// Persisted across navigation/reloads -- a per-browser preference (which db/
// collections/model to chat against), not something that needs to be server-
// authoritative like the ingestion job list did.
const SCOPE_STORAGE_KEY = "edenview.chat.scope";

function loadStoredScope(): ChatScope {
  try {
    const raw = window.localStorage.getItem(SCOPE_STORAGE_KEY);
    return raw ? { ...DEFAULT_SCOPE, ...JSON.parse(raw) } : DEFAULT_SCOPE;
  } catch {
    return DEFAULT_SCOPE;
  }
}

export default function ChatPage() {
  const queryClient = useQueryClient();
  // Always starts from DEFAULT_SCOPE (never localStorage) so the very first render
  // matches on both server and client -- reading localStorage inside a useState
  // initializer caused a hydration mismatch (SSR has no window, so it rendered the
  // "no scope" placeholder text, while the client's first render immediately showed
  // whatever scope was actually saved -- two different texts on the same initial
  // paint). Loading the real value only in an effect below means it only ever
  // happens post-hydration, client-side.
  const [scope, setScope] = useState<ChatScope>(DEFAULT_SCOPE);
  const isFirstSave = useRef(true);

  useEffect(() => {
    setScope(loadStoredScope());
  }, []);

  useEffect(() => {
    // Skip persisting on the very first run -- otherwise this would overwrite the
    // real stored scope with DEFAULT_SCOPE before the load effect above has had a
    // chance to run and replace it.
    if (isFirstSave.current) {
      isFirstSave.current = false;
      return;
    }
    try {
      window.localStorage.setItem(SCOPE_STORAGE_KEY, JSON.stringify(scope));
    } catch {
      // ignore -- e.g. private browsing storage quota
    }
  }, [scope]);
  const [scopePanelOpen, setScopePanelOpen] = useState(true);
  const [sessionListOpen, setSessionListOpen] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [groundingTarget, setGroundingTarget] = useState<GroundingTarget | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const { data: sessionDetail } = useQuery({
    queryKey: ["chat-session", sessionId],
    queryFn: () => getChatSession(sessionId!),
    enabled: !!sessionId,
  });

  useEffect(() => {
    if (sessionDetail) {
      // The server never persists pipelineTrace/thinking (client-only fields, see
      // ChatTurn) -- a naive replace here would wipe them off a turn that was JUST
      // appended locally by onChatSuccess the moment this session's id becomes known
      // for the first time (new-session sends only learn session_id from the
      // response, which triggers this query for the first time on message #1's own
      // turn). Preserve them from the outgoing turns when the same message is still
      // at the same index, so message #1 keeps its trace exactly like #2+ does.
      setTurns((prev) =>
        sessionDetail.messages.map((m, i) => {
          const existing = prev[i];
          const sameMessage = existing?.role === m.role && existing?.content === m.content;
          return {
            role: m.role,
            content: m.content,
            citations: m.citations ?? undefined,
            modelUsed: m.model_used ?? undefined,
            thinking: sameMessage ? existing.thinking : undefined,
            pipelineTrace: sameMessage ? existing.pipelineTrace : undefined,
          };
        }),
      );
    }
  }, [sessionDetail]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const [agenticStatus, setAgenticStatus] = useState<string | null>(null);
  // True only while reattaching to an already-running turn found on session load
  // (see the effect below) -- distinct from agenticMutation.isPending, which only
  // covers a turn started by THIS tab's own send(). Folded into isPending/canSend
  // below so the input/send button/pipeline panel all already gate on it correctly.
  const [reattaching, setReattaching] = useState(false);
  // Live per-node/tool pipeline view -- additive alongside agenticStatus above (that
  // single-line text is untouched). Auto-opens when an agentic turn starts, persists
  // the last completed turn's pipeline until the next one begins (not reset on
  // success/error -- only when a new agentic send() fires). pipelineRef mirrors
  // `pipeline` synchronously (unlike the state, which batches) so onSuccess below can
  // reliably snapshot the FINAL state into the completed turn -- reading `pipeline`
  // directly there risked capturing a stale pre-final-dispatch value depending on
  // React's render timing.
  const [pipeline, setPipeline] = useState<PipelineState>(initialPipelineState);
  const pipelineRef = useRef<PipelineState>(initialPipelineState);
  const [pipelinePanelOpen, setPipelinePanelOpen] = useState(false);
  // Explicit, controlled open/closed state for the LIVE trace disclosure below --
  // deliberately NOT a plain `<details open>` (a constant, not stateful, JSX
  // attribute), since `pipeline` re-renders this component on every streamed event;
  // an uncontrolled `open` would get reasserted on each of those re-renders,
  // silently undoing a user's manual collapse-click moments later.
  const [liveTraceOpen, setLiveTraceOpen] = useState(true);
  const dispatchPipeline = (event: AgenticStatusEvent) => {
    pipelineRef.current = pipelineReducer(pipelineRef.current, event);
    setPipeline(pipelineRef.current);
  };

  const onChatSuccess = (
    res: {
      session_id: string;
      answer: string;
      citations: RetrievalHit[];
      model_used: string;
      thinking?: string | null;
    },
    pipelineTrace?: PipelineState,
  ) => {
    setSessionId(res.session_id);
    setTurns((t) => [
      ...t,
      {
        role: "assistant",
        content: res.answer,
        citations: res.citations,
        modelUsed: res.model_used,
        thinking: res.thinking ?? undefined,
        pipelineTrace,
      },
    ]);
    queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
  };
  const onChatError = (err: Error) => {
    setTurns((t) => [...t, { role: "assistant", content: `Something went wrong: ${err.message}` }]);
  };

  const mutation = useMutation({
    mutationFn: runChat,
    // Explicitly wrapped (not `onSuccess: onChatSuccess`) -- react-query calls
    // onSuccess as (data, variables, context), and passing onChatSuccess directly
    // would make its own `pipelineTrace` param receive react-query's `variables`
    // (the request body) instead of undefined, for Simple RAG turns.
    onSuccess: (res) => onChatSuccess(res),
    onError: onChatError,
  });

  // Only "start" events (and root-level relayed ones) carry a human-readable
  // `message` -- "end" events carry `duration_s` instead, no message. Skip those
  // for this simple status-line display rather than blanking it on every node's
  // completion; a future flowchart-style view would use the full event (node/
  // phase/duration_s) instead of just this derived line.
  const handleAgenticStatus = (event: AgenticStatusEvent) => {
    if (event.message) setAgenticStatus(event.message);
  };

  // Reattaches to a still-running agentic turn when a session is (re)loaded --
  // covers both switching back to a session left mid-turn and a plain page reload,
  // neither of which the old design could recover: live status was only ever
  // driven by the ONE browser tab's own fetch/React state for the ONE connection
  // that started it (confirmed directly: the server-side turn keeps running and
  // persists its answer regardless of whether any client is watching, so a lost
  // connection was never a lost turn -- just a lost VIEW of it). The only
  // reliable, already-available signal that a turn might still be in flight is
  // the persisted messages themselves: the last one is from the user with no
  // assistant reply after it yet (see api/routers/chat.py's GET
  // /chat/stream/{session_id}, which resolves this definitively either way --
  // replaying/continuing a real in-flight turn, or a single cheap "not_running"
  // event if there's nothing to reattach to).
  useEffect(() => {
    if (!sessionDetail || !scope.agentic) return;
    const messages = sessionDetail.messages;
    const lastIsDanglingUser = messages.length > 0 && messages[messages.length - 1].role === "user";
    if (!lastIsDanglingUser) return;

    const controller = new AbortController();
    setReattaching(true);
    pipelineRef.current = initialPipelineState;
    setPipeline(initialPipelineState);
    setPipelinePanelOpen(true);
    setAgenticStatus("Reconnecting…");

    reattachChatStream(
      sessionDetail.session_id,
      (event) => {
        handleAgenticStatus(event);
        dispatchPipeline(event);
      },
      () => setAgenticStatus("Thinking..."),
      controller.signal,
    )
      .then((res) => {
        setAgenticStatus(null);
        if (res) onChatSuccess(res, pipelineRef.current); // null -- nothing was actually in flight, leave turns as-is
      })
      .catch((err) => {
        if (controller.signal.aborted) return; // navigated away again -- not a real error
        setAgenticStatus(null);
        onChatError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => setReattaching(false));

    return () => controller.abort();
    // Deliberately only depends on sessionDetail -- this should re-run when the
    // session's own persisted messages change, not on every render of the
    // (stable-in-practice) callbacks it closes over below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionDetail]);

  // Separate mutation for agentic mode -- POST /chat/stream (see lib/api.ts's
  // runChatStream) surfaces live progress via onStatus while the ADK pipeline runs,
  // since a real turn can genuinely take several minutes and a bare spinner reads
  // as broken for that long. onThinking accumulates the agent's own reasoning
  // narration chunks as they stream in, purely for a "thinking so far..." status
  // line -- the full text ultimately comes back on the result event (res.thinking)
  // and is what actually gets attached to the turn, not this running buffer.
  const agenticMutation = useMutation({
    mutationFn: (body: Parameters<typeof runChatStream>[0]) =>
      runChatStream(
        body,
        (event) => {
          handleAgenticStatus(event);
          dispatchPipeline(event);
        },
        () => setAgenticStatus("Thinking..."),
      ),
    onSuccess: (res) => {
      setAgenticStatus(null);
      onChatSuccess(res, pipelineRef.current);
    },
    onError: (err: Error) => {
      setAgenticStatus(null);
      onChatError(err);
    },
  });

  const isPending = mutation.isPending || agenticMutation.isPending || reattaching;
  const hasScope = scope.collectionNames.length > 0;
  const canSend = hasScope && input.trim().length > 0 && !isPending;

  const handleNewChat = () => {
    setSessionId(null);
    setTurns([]);
    setGroundingTarget(null);
  };

  const handleSelectSession = (id: string) => {
    setGroundingTarget(null);
    setSessionId(id);
  };

  const handleViewSource = (hit: RetrievalHit) => {
    if (hit.bbox == null || hit.page_no == null) return;
    setGroundingTarget({ fileHash: hit.file_hash, pageNo: hit.page_no, bbox: hit.bbox, docStem: hit.doc_stem });
  };

  const send = () => {
    if (!canSend) return;
    const query = input.trim();
    setTurns((t) => [...t, { role: "user", content: query }]);
    setInput("");
    const body = {
      query,
      collection_names: scope.collectionNames,
      top_k: scope.topK,
      use_reranker: scope.useReranker,
      strategy: scope.strategy || undefined,
      chat_model: scope.chatModel || undefined,
      session_id: sessionId ?? undefined,
    };
    if (scope.agentic) {
      setAgenticStatus("Starting...");
      pipelineRef.current = initialPipelineState;
      setPipeline(initialPipelineState);
      setPipelinePanelOpen(true);
      agenticMutation.mutate(body);
    } else {
      mutation.mutate(body);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-1">
      {sessionListOpen ? (
        <ChatSessionList
          activeSessionId={sessionId}
          onSelect={handleSelectSession}
          onNewChat={handleNewChat}
          onCollapse={() => setSessionListOpen(false)}
        />
      ) : (
        <button
          type="button"
          onClick={() => setSessionListOpen(true)}
          title="Show chat history"
          className="flex w-8 shrink-0 items-center justify-center border-r border-border text-muted-foreground hover:bg-muted/40 hover:text-foreground"
        >
          <PanelLeftOpen className="size-4" />
        </button>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex max-w-4xl flex-col gap-6">
            {turns.length === 0 && (
              <div className="flex flex-col items-center gap-2 py-16 text-center text-muted-foreground">
                <Sparkles className="size-6" />
                <p className="text-sm">
                  {hasScope
                    ? "Ask a question about the selected collections."
                    : "Pick a database or collections in chat settings to get started."}
                </p>
                {!scopePanelOpen && (
                  <Button variant="outline" size="sm" onClick={() => setScopePanelOpen(true)}>
                    <PanelRightOpen className="size-3.5" /> Open chat settings
                  </Button>
                )}
              </div>
            )}
            {turns.map((turn, i) => (
              <ChatMessage key={i} turn={turn} onViewSource={handleViewSource} />
            ))}
            {isPending && (
              // Matches ChatMessage's own avatar+bubble layout (same size-7 rounded
              // avatar, same max-w-[80%] column) so the pending state reads as part
              // of the conversation, not a bare status line floating below it.
              <div className="flex gap-3">
                <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  <Sparkles className="size-3.5" />
                </div>
                <div className="flex max-w-[80%] flex-col gap-2">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" /> {agenticMutation.isPending || reattaching ? agenticStatus : "Thinking…"}
                  </div>
                  {/* Live, updating trace (ChatGPT-reasoning-style) while an agentic
                      turn is in progress -- open by default since watching it update
                      in real time is the point, but genuinely collapsible (see
                      liveTraceOpen's own comment for why this can't be a plain
                      `<details open>`). Once the turn finishes, this is replaced by
                      the same content attached to the completed message's own
                      collapsed-by-default disclosure (see chat-message.tsx's
                      pipelineTrace). */}
                  {(agenticMutation.isPending || reattaching) && pipeline !== initialPipelineState && (
                    <details
                      open={liveTraceOpen}
                      className="w-full rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground"
                    >
                      <summary
                        onClick={(e) => {
                          e.preventDefault();
                          setLiveTraceOpen((v) => !v);
                        }}
                        className="flex cursor-pointer list-none items-center gap-1 font-medium text-foreground/80 select-none"
                      >
                        <ChevronRight className={cn("size-3 transition-transform", liveTraceOpen && "rotate-90")} />
                        Reasoning trace
                      </summary>
                      <div className="mt-1.5">
                        <AgentTrace state={pipeline} />
                      </div>
                    </details>
                  )}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-border px-6 py-4">
          <div className="mx-auto flex max-w-4xl items-end gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder={hasScope ? "Ask a question…" : "Select a scope first…"}
              disabled={!hasScope}
              className="min-h-11 resize-none"
              rows={1}
            />
            <Button size="icon" disabled={!canSend} onClick={send}>
              <Send className="size-4" />
            </Button>
          </div>
        </div>
      </div>

      {scopePanelOpen ? (
        <ChatScopePanel scope={scope} onChange={setScope} onCollapse={() => setScopePanelOpen(false)} />
      ) : (
        <button
          type="button"
          onClick={() => setScopePanelOpen(true)}
          title="Show chat settings"
          className="flex w-8 shrink-0 items-center justify-center border-l border-border text-muted-foreground hover:bg-muted/40 hover:text-foreground"
        >
          <PanelRightOpen className="size-4" />
        </button>
      )}

      {pipelinePanelOpen ? (
        <AgentPipelinePanel state={pipeline} onCollapse={() => setPipelinePanelOpen(false)} />
      ) : (
        pipeline !== initialPipelineState && (
          <button
            type="button"
            onClick={() => setPipelinePanelOpen(true)}
            title="Show agent pipeline"
            className="flex w-8 shrink-0 items-center justify-center border-l border-border text-muted-foreground hover:bg-muted/40 hover:text-foreground"
          >
            <PanelRightOpen className="size-4" />
          </button>
        )
      )}

      {groundingTarget && <GroundingPanel target={groundingTarget} onClose={() => setGroundingTarget(null)} />}
    </div>
  );
}
