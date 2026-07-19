"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getChatSession, runChat, runChatStream } from "@/lib/api";
import { ChatScopePanel, type ChatScope } from "@/components/chat/chat-scope-panel";
import { ChatMessage, type ChatTurn } from "@/components/chat/chat-message";
import { ChatSessionList } from "@/components/chat/chat-session-list";
import { GroundingPanel, type GroundingTarget } from "@/components/chat/grounding-panel";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import type { RetrievalHit } from "@/lib/types";
import { Send, Loader2, Sparkles, PanelRightOpen } from "lucide-react";

const DEFAULT_SCOPE: ChatScope = {
  collectionNames: [],
  topK: 5,
  useReranker: true,
  strategy: "",
  chatModel: "",
  agentic: false,
  effort: "high",
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
      setTurns(
        sessionDetail.messages.map((m) => ({
          role: m.role,
          content: m.content,
          citations: m.citations ?? undefined,
          modelUsed: m.model_used ?? undefined,
        })),
      );
    }
  }, [sessionDetail]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const [agenticStatus, setAgenticStatus] = useState<string | null>(null);

  const onChatSuccess = (res: {
    session_id: string;
    answer: string;
    citations: RetrievalHit[];
    model_used: string;
    thinking?: string | null;
  }) => {
    setSessionId(res.session_id);
    setTurns((t) => [
      ...t,
      { role: "assistant", content: res.answer, citations: res.citations, modelUsed: res.model_used, thinking: res.thinking ?? undefined },
    ]);
    queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
  };
  const onChatError = (err: Error) => {
    setTurns((t) => [...t, { role: "assistant", content: `Something went wrong: ${err.message}` }]);
  };

  const mutation = useMutation({
    mutationFn: runChat,
    onSuccess: onChatSuccess,
    onError: onChatError,
  });

  // Separate mutation for agentic mode -- POST /chat/stream (see lib/api.ts's
  // runChatStream) surfaces live progress via onStatus while ADK's agent loop runs,
  // since a "high" effort turn can genuinely take 30-90+ seconds and a bare spinner
  // reads as broken for that long. onThinking accumulates the agent's own reasoning
  // narration chunks as they stream in, purely for a "thinking so far..." status
  // line -- the full text ultimately comes back on the result event (res.thinking)
  // and is what actually gets attached to the turn, not this running buffer.
  const agenticMutation = useMutation({
    mutationFn: (body: Parameters<typeof runChatStream>[0]) =>
      runChatStream(body, setAgenticStatus, () => setAgenticStatus("Thinking...")),
    onSuccess: (res) => {
      setAgenticStatus(null);
      onChatSuccess(res);
    },
    onError: (err: Error) => {
      setAgenticStatus(null);
      onChatError(err);
    },
  });

  const isPending = mutation.isPending || agenticMutation.isPending;
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
      agenticMutation.mutate({ ...body, effort: scope.effort });
    } else {
      mutation.mutate(body);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-1">
      <ChatSessionList activeSessionId={sessionId} onSelect={handleSelectSession} onNewChat={handleNewChat} />

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex max-w-3xl flex-col gap-6">
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
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" /> {agenticMutation.isPending ? agenticStatus : "Thinking…"}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-border px-6 py-4">
          <div className="mx-auto flex max-w-3xl items-end gap-2">
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

      {groundingTarget && <GroundingPanel target={groundingTarget} onClose={() => setGroundingTarget(null)} />}
    </div>
  );
}
