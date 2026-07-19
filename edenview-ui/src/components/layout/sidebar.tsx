"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { UploadCloud, Database, MessageSquare, Settings, Circle, Cpu } from "lucide-react";
import { getSystemInfo } from "@/lib/api";
import { SystemMonitor } from "@/components/layout/system-monitor";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/ingestion", label: "Ingestion", icon: UploadCloud },
  { href: "/collections", label: "Collections", icon: Database },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/settings", label: "Settings", icon: Settings },
];

// Single source of truth for the footer's company name -- likely to change, keep it
// a one-line edit rather than a string repeated anywhere else.
const COMPANY_NAME = "EdenLabs";

export function Sidebar() {
  const pathname = usePathname();
  const { data: system } = useQuery({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    refetchInterval: 30_000,
  });

  const ollamaUp = system?.ollama.available ?? null;
  const gpu = system?.gpus[0];

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground text-sm font-semibold">
          E
        </div>
        <span className="text-[15px] font-semibold tracking-tight">Edenview</span>
      </div>

      <nav className="flex flex-col gap-0.5 px-3">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href || pathname?.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
              )}
            >
              <Icon className="size-4" strokeWidth={2} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto flex flex-col gap-2 border-t border-sidebar-border px-4 py-4 text-xs text-sidebar-foreground/60">
        <div className="flex items-center gap-2">
          <Circle
            className={cn(
              "size-2",
              ollamaUp === null ? "fill-muted-foreground text-muted-foreground" : ollamaUp ? "fill-emerald-500 text-emerald-500" : "fill-red-500 text-red-500",
            )}
          />
          <span>{ollamaUp === null ? "Checking Ollama…" : ollamaUp ? "Ollama connected" : "Ollama unreachable"}</span>
        </div>
        {gpu && (
          <div className="flex items-center gap-2">
            <Cpu className="size-3.5" />
            <span className="truncate" title={gpu.name}>
              {gpu.name}
            </span>
          </div>
        )}
      </div>
      <SystemMonitor />
      <div className="border-t border-sidebar-border px-4 py-2 text-center text-sidebar-foreground/40">
        <p className="text-[10px]">
          © {new Date().getFullYear()} {COMPANY_NAME}
        </p>
        <p className="text-[9px]">Developed for Everyone by David Babu</p>
      </div>
    </aside>
  );
}
