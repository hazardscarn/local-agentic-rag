"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { deleteChatSession, listChatSessions } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { cn } from "@/lib/utils";
import { Plus, Trash2, Loader2, MessageSquareText } from "lucide-react";

function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function ChatSessionList({
  activeSessionId,
  onSelect,
  onNewChat,
}: {
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
}) {
  const queryClient = useQueryClient();
  const { data: sessions, isLoading } = useQuery({ queryKey: ["chat-sessions"], queryFn: listChatSessions });

  const deleteMutation = useMutation({
    mutationFn: deleteChatSession,
    onSuccess: (_void, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      if (sessionId === activeSessionId) onNewChat();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col gap-3 border-r border-border px-3 py-5">
      <Button variant="outline" size="sm" className="justify-start" onClick={onNewChat}>
        <Plus className="size-3.5" /> New chat
      </Button>

      <div className="flex flex-col gap-0.5 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center gap-2 px-2 py-2 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> Loading…
          </div>
        )}
        {sessions?.length === 0 && (
          <p className="px-2 py-2 text-xs text-muted-foreground">No conversations yet.</p>
        )}
        {sessions?.map((session) => (
          <div
            key={session.session_id}
            className={cn(
              "group flex items-center gap-1 rounded-lg px-2 py-2 text-left text-sm",
              session.session_id === activeSessionId ? "bg-accent/60 text-foreground" : "hover:bg-muted/60",
            )}
          >
            <button
              type="button"
              onClick={() => onSelect(session.session_id)}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              <MessageSquareText className="size-3.5 shrink-0 text-muted-foreground" />
              <span className="flex min-w-0 flex-col">
                <span className="truncate">{session.title}</span>
                <span className="text-[11px] text-muted-foreground">{relativeTime(session.updated_at)}</span>
              </span>
            </button>
            <ConfirmDialog
              trigger={
                <button
                  type="button"
                  className="invisible shrink-0 rounded p-1 text-muted-foreground hover:text-destructive group-hover:visible"
                >
                  <Trash2 className="size-3.5" />
                </button>
              }
              title="Delete this conversation?"
              description={`"${session.title}" and all its messages will be permanently removed.`}
              onConfirm={() => deleteMutation.mutate(session.session_id)}
              isPending={deleteMutation.isPending}
            />
          </div>
        ))}
      </div>
    </aside>
  );
}
