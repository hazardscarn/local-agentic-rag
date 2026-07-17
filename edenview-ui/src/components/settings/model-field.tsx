"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { RESTART_REQUIRED_KEYS } from "@/lib/types";
import type { OllamaModelInfo } from "@/lib/types";

export function ModelField({
  settingKey,
  label,
  value,
  onChange,
  pulledModels,
  hint,
}: {
  settingKey: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  pulledModels?: OllamaModelInfo[];
  hint?: string;
}) {
  const restartRequired = RESTART_REQUIRED_KEYS.has(settingKey);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <Label htmlFor={settingKey}>{label}</Label>
        <Badge variant={restartRequired ? "secondary" : "outline"} className="text-[10px]">
          {restartRequired ? "restart required" : "applies immediately"}
        </Badge>
      </div>
      <Input id={settingKey} value={value} onChange={(e) => onChange(e.target.value)} />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {pulledModels && pulledModels.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {pulledModels.map((m) => (
            <button
              key={m.name}
              type="button"
              onClick={() => onChange(m.name)}
              className="rounded-full border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:border-primary hover:text-foreground"
            >
              {m.name} <span className="text-muted-foreground/70">{m.size_gb}GB</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
