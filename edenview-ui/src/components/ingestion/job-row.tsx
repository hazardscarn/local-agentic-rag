"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ChunkPreviewList } from "@/components/shared/chunk-preview-list";
import { StageStepper } from "@/components/ingestion/stage-stepper";
import { retryJob, cancelJob, ApiError } from "@/lib/api";
import type { IngestionJobRecord } from "@/lib/types";
import { ChevronDown, ChevronUp, AlertCircle, RotateCcw, Loader2, X } from "lucide-react";

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  queued: "secondary",
  running: "default",
  done: "outline",
  error: "destructive",
  cancelled: "secondary",
};

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function JobRow({ job }: { job: IngestionJobRecord }) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const queryClient = useQueryClient();

  const isDone = job.status === "done";
  const isError = job.status === "error";
  const isCancelled = job.status === "cancelled";
  const isActive = job.status === "queued" || job.status === "running";

  const startedAt = job.started_at ? new Date(job.started_at).getTime() : null;
  const finishedAt = job.finished_at ? new Date(job.finished_at).getTime() : null;
  const durationMs = startedAt !== null ? (finishedAt ?? Date.now()) - startedAt : null;

  const retryMutation = useMutation({
    mutationFn: () => retryJob(job.job_id),
    onSuccess: () => {
      toast.success(`Retrying ${job.filename ?? "job"}`);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (err) => {
      toast.error(err instanceof ApiError ? err.message : "Retry failed");
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(job.job_id),
    onSuccess: () => {
      toast.success(`Cancelling ${job.filename ?? "job"}…`);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (err) => {
      toast.error(err instanceof ApiError ? err.message : "Cancel failed");
    },
  });

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{job.filename ?? "(unnamed file)"}</p>
          <p className="text-xs text-muted-foreground">
            {job.db_name ?? "?"} / {job.qdrant_collection_name ?? "?"}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Badge variant={STATUS_VARIANT[job.status]}>{job.status}</Badge>
          {durationMs !== null && (
            <span className="text-[11px] text-muted-foreground">
              {isActive ? `${formatDuration(durationMs)} elapsed` : formatDuration(durationMs)}
            </span>
          )}
        </div>
      </div>

      {isActive && (
        <div className="mt-4">
          <StageStepper job={job} />
          <div className="mt-3 flex justify-end">
            <Button
              variant="outline"
              size="sm"
              disabled={cancelMutation.isPending}
              onClick={() => cancelMutation.mutate()}
            >
              {cancelMutation.isPending ? <Loader2 className="size-3.5 animate-spin" /> : <X className="size-3.5" />}
              Cancel
            </Button>
          </div>
        </div>
      )}

      {isCancelled && (
        <p className="mt-3 text-sm text-muted-foreground">{job.error_message ?? "Cancelled by user."}</p>
      )}

      {isError && (
        <Alert variant="destructive" className="mt-3">
          <AlertCircle className="size-4" />
          <AlertDescription className="flex flex-1 items-center justify-between gap-3">
            <span>{job.error_message ?? "Ingestion failed."}</span>
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              disabled={retryMutation.isPending}
              onClick={() => retryMutation.mutate()}
            >
              {retryMutation.isPending ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <RotateCcw className="size-3.5" />
              )}
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {isDone && job.qdrant_collection_name && (
        <div className="mt-3">
          <Button variant="outline" size="sm" onClick={() => setPreviewOpen((v) => !v)}>
            {previewOpen ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
            Preview chunks
          </Button>
          {previewOpen && (
            <div className="mt-3">
              <ChunkPreviewList collectionName={job.qdrant_collection_name} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
