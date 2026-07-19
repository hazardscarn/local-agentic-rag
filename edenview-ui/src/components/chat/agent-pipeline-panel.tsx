"use client";

import { PanelRightClose } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AgentFlowchart } from "@/components/chat/agent-flowchart";
import type { PipelineState } from "@/lib/pipeline-state";

export function AgentPipelinePanel({ state, onCollapse }: { state: PipelineState; onCollapse: () => void }) {
  const activeThread = state.threads[state.threads.length - 1];
  const isEmpty = !state.decompose && state.threads.length === 0 && !state.answerFormatter;

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-border px-4 py-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Agent pipeline</span>
        <Button variant="ghost" size="icon" className="size-7" onClick={onCollapse} title="Hide agent pipeline">
          <PanelRightClose className="size-4" />
        </Button>
      </div>

      {isEmpty ? (
        <p className="text-xs text-muted-foreground">{state.rootMessage ?? "Waiting to start…"}</p>
      ) : (
        <>
          {activeThread && (
            <p className="text-xs text-muted-foreground">
              {activeThread.total > 1
                ? `Researching (${activeThread.index}/${activeThread.total}): ${activeThread.text}`
                : `Researching: ${activeThread.text}`}
            </p>
          )}
          <AgentFlowchart state={state} />
        </>
      )}
    </aside>
  );
}
