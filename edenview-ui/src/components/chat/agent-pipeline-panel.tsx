"use client";

import { PanelRightClose } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AgentFlowchart } from "@/components/chat/agent-flowchart";
import type { PipelineState } from "@/lib/pipeline-state";

export function AgentPipelinePanel({ state, onCollapse }: { state: PipelineState; onCollapse: () => void }) {
  const activeThread = state.threads[state.threads.length - 1];

  // NEW architecture: researcher/rootAgent take priority for the active panel check
  const isActive = !!state.researcher || !!state.rootAgent || state.threads.length > 0 || state.decompose || state.answerFormatter;

  return (
    <aside className="flex h-full w-[26rem] shrink-0 flex-col gap-3 overflow-y-auto border-l border-border px-4 py-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Agent pipeline</span>
        <Button variant="ghost" size="icon" className="size-7" onClick={onCollapse} title="Hide agent pipeline">
          <PanelRightClose className="size-4" />
        </Button>
      </div>

      {!isActive ? (
        <p className="text-xs text-muted-foreground">{state.rootMessage ?? "Waiting to start…"}</p>
      ) : (
        <>
          {/* NEW architecture: show researcher phase */}
          {state.researcher && state.researcher.status === "running" && !activeThread && (
            <p className="text-xs text-muted-foreground">Researching…</p>
          )}
          {/* Legacy: show sub-question context when present */}
          {activeThread && activeThread.total > 1 ? (
            <p className="text-xs text-muted-foreground">Researching ({activeThread.index}/{activeThread.total}): {activeThread.text}</p>
          ) : activeThread ? (
            <p className="text-xs text-muted-foreground">Researching: {activeThread.text}</p>
          ) : null}
          <AgentFlowchart state={state} />
        </>
      )}
    </aside>
  );
}
