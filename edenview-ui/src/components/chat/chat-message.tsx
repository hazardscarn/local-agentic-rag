"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { CitationCard } from "@/components/chat/citation-card";
import { AgentTrace } from "@/components/chat/agent-trace";
import type { RetrievalHit } from "@/lib/types";
import type { PipelineState } from "@/lib/pipeline-state";
import { cn } from "@/lib/utils";
import { ChevronRight, Sparkles, User } from "lucide-react";

// Turns literal "[1]"/"[2]" citation markers (edenview_RAG/agentic_rag's
// ANSWER_FORMATTER_INSTRUCTION convention -- digit-only bracket markers,
// nothing else uses this exact shape) into markdown links ReactMarkdown will
// render as real <a> elements, so the custom `a` renderer below (via
// `components`) can intercept them and make them clickable instead of just
// plain inline text. Adjacent markers like "[2][3]" become two independent
// links, unaffected. Requires runtime.py's `_ordered_citations` to have
// numbered `citations[]` in the SAME order these markers use (via
// state["final_citation_order"], see callbacks.prepare_consolidated_findings)
// -- without that, marker N and citations[N-1] would be unrelated chunks.
const CITATION_MARKER_RE = /\[(\d+)\]/g;

function linkifyCitations(content: string): string {
  return content.replace(CITATION_MARKER_RE, (match, n) => `[${match}](#cite-${n})`);
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  citations?: RetrievalHit[];
  modelUsed?: string;
  // Agentic mode only -- the agent's own reasoning/planning narration for this
  // turn. Shown collapsed by default via a <details> disclosure, never as part of
  // `content` itself (see runtime.py's _extract_final_text/_extract_thought_text
  // split -- this is what fixed reasoning text leaking into the real answer).
  thinking?: string;
  // Agentic mode only -- a snapshot of the live pipeline state (chat/page.tsx's
  // `pipeline`) captured at the moment this turn finished, so the step-by-step
  // trace (decompose/per-sub-question timelines/answer) stays viewable after the
  // fact, ChatGPT-reasoning-style, even though the live view resets on the next turn.
  pipelineTrace?: PipelineState;
}

export function ChatMessage({ turn, onViewSource }: { turn: ChatTurn; onViewSource?: (hit: RetrievalHit) => void }) {
  const isUser = turn.role === "user";
  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex size-7 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-secondary text-secondary-foreground" : "bg-primary text-primary-foreground",
        )}
      >
        {isUser ? <User className="size-3.5" /> : <Sparkles className="size-3.5" />}
      </div>
      <div className={cn("flex max-w-[80%] flex-col gap-2", isUser && "items-end")}>
        {!isUser && turn.thinking && (
          <details className="group w-full rounded-lg border border-border/60 bg-muted/30 px-3 py-1.5 text-xs text-muted-foreground">
            <summary className="flex cursor-pointer list-none items-center gap-1 select-none">
              <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
              Thinking
            </summary>
            <p className="mt-1.5 whitespace-pre-wrap leading-relaxed">{turn.thinking}</p>
          </details>
        )}
        {!isUser && turn.pipelineTrace && (
          <details className="group w-full rounded-lg border border-border/60 bg-muted/30 px-3 py-1.5 text-xs">
            <summary className="flex cursor-pointer list-none items-center gap-1 select-none text-muted-foreground">
              <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
              Show reasoning trace
            </summary>
            <div className="mt-2">
              <AgentTrace state={turn.pipelineTrace} />
            </div>
          </details>
        )}
        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser ? "bg-primary text-primary-foreground text-sm" : "bg-card border border-border",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{turn.content}</p>
          ) : (
            <div className="prose dark:prose-invert max-w-none prose-p:my-3 prose-p:leading-relaxed prose-headings:mt-4 prose-headings:mb-2 prose-ul:my-3 prose-ol:my-3 prose-li:my-1 prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none prose-table:block prose-table:overflow-x-auto prose-th:text-left prose-td:align-top">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={{
                  a: ({ href, children, ...props }) => {
                    const citeMatch = href?.match(/^#cite-(\d+)$/);
                    if (!citeMatch) {
                      return (
                        <a href={href} target="_blank" rel="noreferrer" {...props}>
                          {children}
                        </a>
                      );
                    }
                    const hit = turn.citations?.[Number(citeMatch[1]) - 1];
                    const canGround = hit != null && hit.bbox != null && hit.page_no != null;
                    if (!canGround) {
                      // No source to jump to (ungroundable hit, or the marker
                      // outran the citations array) -- render as plain styled
                      // text, matching CitationCard's own behavior for the
                      // same case, not a dead link.
                      return (
                        <span className="mx-0.5 rounded bg-primary/15 px-1 text-[0.7em] font-semibold text-primary no-underline">
                          {children}
                        </span>
                      );
                    }
                    return (
                      <button
                        type="button"
                        onClick={() => onViewSource?.(hit)}
                        title={hit.doc_stem}
                        className="mx-0.5 cursor-pointer rounded bg-primary/15 px-1 text-[0.7em] font-semibold text-primary no-underline hover:bg-primary/25"
                      >
                        {children}
                      </button>
                    );
                  },
                }}
              >
                {linkifyCitations(turn.content)}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {turn.citations && turn.citations.length > 0 && (
          <div className="flex w-full gap-2 overflow-x-auto pb-1">
            {turn.citations.map((hit, i) => (
              <CitationCard key={hit.chunk_id} index={i + 1} hit={hit} onViewSource={onViewSource} />
            ))}
          </div>
        )}
        {turn.modelUsed && <span className="text-[11px] text-muted-foreground">{turn.modelUsed}</span>}
      </div>
    </div>
  );
}
