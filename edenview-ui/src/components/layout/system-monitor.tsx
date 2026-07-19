"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { getSystemInfo, unloadOllamaModel } from "@/lib/api";
import { AreaChart } from "@/components/layout/area-chart";
import type { LoadedModelInfo } from "@/lib/types";
import { MemoryStick, Loader2, X, ChevronDown, ChevronUp, Activity } from "lucide-react";

// 15 samples @ 4s = a clean 60s rolling window.
const HISTORY_LENGTH = 15;
const POLL_INTERVAL_MS = 4000;
const WINDOW_SECONDS = (HISTORY_LENGTH * POLL_INTERVAL_MS) / 1000;

function minutesUntil(iso: string | null): string | null {
  if (!iso) return null;
  const mins = Math.round((new Date(iso).getTime() - Date.now()) / 60_000);
  if (mins <= 0) return "evicting…";
  return `${mins}m left`;
}

// Ollama has no separate "% on GPU" field -- it's derived the same way `ollama ps`
// derives its own CPU/GPU split column: size_vram_gb is the slice of the model's
// total size that made it into VRAM, so the rest is necessarily running on CPU.
function gpuSplit(m: LoadedModelInfo): { gpuPct: number; cpuPct: number } | null {
  if (!m.size_gb) return null;
  const gpuPct = Math.max(0, Math.min(100, Math.round((m.size_vram_gb / m.size_gb) * 100)));
  return { gpuPct, cpuPct: 100 - gpuPct };
}

function gpuSplitLabel(split: { gpuPct: number; cpuPct: number }): string {
  if (split.gpuPct === 100) return "100% GPU";
  if (split.cpuPct === 100) return "100% CPU";
  return `${split.gpuPct}% GPU / ${split.cpuPct}% CPU`;
}

export function SystemMonitor() {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(true);
  const [ramHistory, setRamHistory] = useState<number[]>([]);
  const [vramHistory, setVramHistory] = useState<number[]>([]);

  const { data } = useQuery({
    queryKey: ["system-live"],
    queryFn: getSystemInfo,
    refetchInterval: POLL_INTERVAL_MS,
  });

  const unloadMutation = useMutation({
    mutationFn: unloadOllamaModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["system-live"] }),
    onError: (err: Error) => toast.error(err.message),
  });

  const gpu = data?.gpus[0];
  const ramUsedGb = data ? data.ram_total_gb - data.ram_available_gb : null;
  const ramUsedPct = data ? Math.round((ramUsedGb! / data.ram_total_gb) * 100) : null;
  const vramUsedMb = gpu?.vram_total_mb != null && gpu.vram_free_mb != null ? gpu.vram_total_mb - gpu.vram_free_mb : null;
  const vramUsedPct = gpu?.vram_total_mb && vramUsedMb != null ? Math.round((vramUsedMb / gpu.vram_total_mb) * 100) : null;

  useEffect(() => {
    if (ramUsedPct == null) return;
    setRamHistory((h) => [...h, ramUsedPct].slice(-HISTORY_LENGTH));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only append on a genuinely new sample
  }, [ramUsedPct]);

  useEffect(() => {
    if (vramUsedPct == null) return;
    setVramHistory((h) => [...h, vramUsedPct].slice(-HISTORY_LENGTH));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only append on a genuinely new sample
  }, [vramUsedPct]);

  if (!data) {
    return (
      <div className="flex items-center gap-2 px-4 py-3 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" /> Reading system…
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 border-t border-sidebar-border px-4 py-3">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center justify-between text-[11px] font-medium text-sidebar-foreground/70 hover:text-sidebar-foreground"
      >
        <span className="flex items-center gap-1.5">
          <Activity className="size-3.5" /> Performance
        </span>
        {expanded ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
      </button>

      {expanded && (
        <>
          <AreaChart
            label="RAM"
            values={ramHistory}
            currentLabel={`${ramUsedGb?.toFixed(1)}/${data.ram_total_gb}GB`}
            colorClassName="text-[#2a78d6] dark:text-[#3987e5]"
            gradientId="ram-gradient"
            windowSeconds={WINDOW_SECONDS}
          />
          {gpu?.vram_total_mb != null && (
            <AreaChart
              label="VRAM"
              values={vramHistory}
              currentLabel={`${(vramUsedMb! / 1024).toFixed(1)}/${(gpu.vram_total_mb / 1024).toFixed(1)}GB`}
              colorClassName="text-[#008300]"
              gradientId="vram-gradient"
              windowSeconds={WINDOW_SECONDS}
            />
          )}

          {data.loaded_models.length > 0 && (
            <div className="flex flex-col gap-1">
              <p className="flex items-center gap-1 text-[11px] font-medium text-sidebar-foreground/70">
                <MemoryStick className="size-3" /> Loaded in Ollama
              </p>
              {data.loaded_models.map((m) => {
                const split = gpuSplit(m);
                return (
                  <div key={m.name} className="flex flex-col gap-1">
                    <div className="flex items-center justify-between gap-1 text-[11px] text-sidebar-foreground/60">
                      <span className="truncate" title={`${m.name} · ${m.size_vram_gb}GB VRAM · ${minutesUntil(m.expires_at) ?? ""}`}>
                        {m.name} · {m.size_vram_gb}GB
                      </span>
                      <button
                        type="button"
                        onClick={() => unloadMutation.mutate(m.name)}
                        disabled={unloadMutation.isPending}
                        className="shrink-0 rounded p-0.5 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground disabled:opacity-50"
                        title="Unload from memory now"
                      >
                        <X className="size-3" />
                      </button>
                    </div>
                    {split && (
                      <div className="flex items-center gap-1.5 pl-0.5" title="Share of this model's memory resident in VRAM vs. spilled to system RAM">
                        <div className="flex h-1 w-full overflow-hidden rounded-full">
                          {split.gpuPct > 0 && (
                            <div
                              className="h-full rounded-full bg-[#008300] dark:bg-[#00a300]"
                              style={{ width: `${split.gpuPct}%` }}
                            />
                          )}
                          {split.gpuPct > 0 && split.cpuPct > 0 && <div className="w-[2px]" />}
                          {split.cpuPct > 0 && (
                            <div
                              className="h-full rounded-full bg-[#2a78d6] dark:bg-[#3987e5]"
                              style={{ width: `${split.cpuPct}%` }}
                            />
                          )}
                        </div>
                        <span className="shrink-0 text-[9px] tabular-nums text-sidebar-foreground/40">
                          {gpuSplitLabel(split)}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
