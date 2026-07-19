"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { CitationCard } from "@/components/chat/citation-card";
import type { RetrievalHit } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ChevronRight, Sparkles, User } from "lucide-react";

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
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                {turn.content}
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
