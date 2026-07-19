"use client";

import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

export function CollectionStatusBadge({ status }: { status: string }) {
  if (status === "ingesting") {
    return (
      <Badge variant="secondary" className="gap-1">
        <Loader2 className="size-3 animate-spin" /> ingesting
      </Badge>
    );
  }
  if (status === "error") {
    return <Badge variant="destructive">error</Badge>;
  }
  return <Badge variant="outline">{status}</Badge>;
}
