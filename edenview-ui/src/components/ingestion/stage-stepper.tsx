"use client";

import { Check, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { IngestionJobRecord } from "@/lib/types";

const STAGES: { key: IngestionJobRecord["stage"] & string; label: string }[] = [
  { key: "extracting", label: "Extract" },
  { key: "chunking", label: "Chunk" },
  { key: "embedding", label: "Embed" },
];

type StepState = "done" | "current" | "pending" | "error";

function stepState(index: number, job: IngestionJobRecord): StepState {
  const stageIndex = STAGES.findIndex((s) => s.key === job.stage);
  if (job.status === "done") return "done";
  if (job.status === "error") {
    if (stageIndex === -1 || index < stageIndex) return "done";
    if (index === stageIndex) return "error";
    return "pending";
  }
  if (job.status === "queued" || stageIndex === -1) return "pending";
  if (index < stageIndex) return "done";
  if (index === stageIndex) return "current";
  return "pending";
}

export function StageStepper({ job }: { job: IngestionJobRecord }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center">
        {STAGES.map((stage, i) => {
          const state = stepState(i, job);
          return (
            <div key={stage.key} className="flex flex-1 items-center last:flex-none">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={cn(
                    "flex size-6 items-center justify-center rounded-full border text-[11px] font-medium",
                    state === "done" && "border-primary bg-primary text-primary-foreground",
                    state === "current" && "border-primary text-primary",
                    state === "pending" && "border-border text-muted-foreground",
                    state === "error" && "border-destructive bg-destructive/10 text-destructive",
                  )}
                >
                  {state === "done" && <Check className="size-3.5" />}
                  {state === "current" && <Loader2 className="size-3.5 animate-spin" />}
                  {state === "error" && <X className="size-3.5" />}
                  {state === "pending" && i + 1}
                </div>
                <span
                  className={cn(
                    "text-[11px]",
                    state === "pending" ? "text-muted-foreground" : "text-foreground",
                  )}
                >
                  {stage.label}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <div
                  className={cn(
                    "mx-1.5 mb-4 h-px flex-1",
                    stepState(i + 1, job) === "pending" ? "bg-border" : "bg-primary",
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
      {job.stage === "embedding" && job.stage_pct != null && (
        <p className="text-center text-xs text-muted-foreground">{job.stage_pct}% embedded</p>
      )}
      {job.status !== "done" && job.status !== "error" && (
        <p className="text-center text-xs text-muted-foreground">
          Larger documents can take a few minutes — feel free to check back later.
        </p>
      )}
    </div>
  );
}
