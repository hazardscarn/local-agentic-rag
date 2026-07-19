"use client";

import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RESTART_REQUIRED_KEYS } from "@/lib/types";
import type { OllamaModelCapabilities } from "@/lib/types";

// Unlike ModelField (free-text input + suggestion pills -- can't hard-enforce a
// choice), this is a real <Select> filtered to only models that report the given
// capability, since picking an incompatible one isn't just suboptimal here, it's a
// hard failure (e.g. agent_model without "tools" support crashes the agentic RAG
// pipeline at import time -- see require_tool_calling_model()).
export function AgentModelSelect({
  settingKey,
  label,
  value,
  onChange,
  capability,
  models,
  allowUnset,
  unsetLabel,
}: {
  settingKey: string;
  label: string;
  value: string; // "" means unset when allowUnset
  onChange: (v: string) => void;
  capability: "tools" | "vision";
  models: OllamaModelCapabilities[];
  allowUnset?: boolean;
  unsetLabel?: string;
}) {
  const restartRequired = RESTART_REQUIRED_KEYS.has(settingKey);
  const compatible = models.filter((m) => m.capabilities.includes(capability));

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <Label htmlFor={settingKey}>{label}</Label>
        <Badge variant={restartRequired ? "secondary" : "outline"} className="text-[10px]">
          {restartRequired ? "restart required" : "applies immediately"}
        </Badge>
      </div>
      {compatible.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No pulled model reports &quot;{capability}&quot; support -- pull one first (e.g. {" "}
          <code className="rounded bg-muted px-1 py-0.5">ollama pull qwen3.5:4b</code>).
        </p>
      ) : (
        <Select
          value={value || (allowUnset ? "__default__" : undefined)}
          onValueChange={(v) => onChange(v === "__default__" || !v ? "" : v)}
        >
          <SelectTrigger id={settingKey} className="w-full">
            <SelectValue>{(v: string | null) => (!v || v === "__default__" ? (unsetLabel ?? "Use default") : v)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {allowUnset && <SelectItem value="__default__">{unsetLabel ?? "Use default"}</SelectItem>}
            {compatible.map((m) => (
              <SelectItem key={m.name} value={m.name}>
                {m.name} · {m.size_gb} GB
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
