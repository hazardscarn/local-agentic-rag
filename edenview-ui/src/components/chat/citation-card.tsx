"use client";

import { fileUrl } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import type { RetrievalHit } from "@/lib/types";
import { ImageIcon, ScanSearch } from "lucide-react";

export function CitationCard({
  index,
  hit,
  onViewSource,
}: {
  index: number;
  hit: RetrievalHit;
  onViewSource?: (hit: RetrievalHit) => void;
}) {
  // Grounding needs both a page number and a bbox -- recursive_overlap hits and
  // anything ingested before visual grounding was added have neither.
  const canGround = hit.bbox != null && hit.page_no != null;

  return (
    <div className="flex w-64 shrink-0 flex-col gap-1.5 rounded-lg border border-border bg-card p-3">
      <div className="flex items-center gap-1.5">
        <span className="flex size-4 items-center justify-center rounded bg-primary/15 text-[10px] font-semibold text-primary">
          {index}
        </span>
        <span className="truncate text-xs font-medium" title={hit.doc_stem}>
          {hit.doc_stem}
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {hit.page_no != null && (
          <Badge variant="secondary" className="text-[10px]">
            page {hit.page_no}
          </Badge>
        )}
        <Badge variant="outline" className="text-[10px]">
          {hit.strategy}
        </Badge>
      </div>
      <p className="line-clamp-4 text-xs text-muted-foreground">{hit.text}</p>
      <div className="flex items-center gap-3">
        {canGround && onViewSource && (
          <button
            type="button"
            onClick={() => onViewSource(hit)}
            className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground"
          >
            <ScanSearch className="size-3" /> view source page
          </button>
        )}
        {hit.images.length > 0 && (
          <a
            href={fileUrl(hit.images[0].image_path)}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground"
          >
            <ImageIcon className="size-3" /> view image
          </a>
        )}
      </div>
    </div>
  );
}
