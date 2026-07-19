"use client";

import { useState } from "react";
import { documentPageUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import type { Bbox } from "@/lib/types";
import { X, Maximize2 } from "lucide-react";

export interface GroundingTarget {
  fileHash: string;
  pageNo: number;
  bbox: Bbox | null;
  docStem: string;
}

export function GroundingPanel({ target, onClose }: { target: GroundingTarget; onClose: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const src = documentPageUrl(target.fileHash, target.pageNo, target.bbox);
  const alt = `${target.docStem} page ${target.pageNo}`;

  return (
    <aside className="flex h-full w-[28rem] shrink-0 flex-col border-l border-border">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium" title={target.docStem}>
            {target.docStem}
          </p>
          <p className="text-xs text-muted-foreground">page {target.pageNo}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="icon" className="size-7" onClick={() => setExpanded(true)} title="Expand">
            <Maximize2 className="size-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={onClose}>
            <X className="size-3.5" />
          </Button>
        </div>
      </div>
      <div className="flex-1 overflow-auto bg-muted/30 p-3">
        {/* eslint-disable-next-line @next/next/no-img-element -- rendered on demand by the backend (pypdfium2), not a static/optimizable asset */}
        <img
          src={src}
          alt={alt}
          className="mx-auto w-full cursor-zoom-in rounded-md border border-border shadow-sm"
          onClick={() => setExpanded(true)}
        />
      </div>

      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="flex max-h-[95vh] w-full max-w-[95vw] flex-col gap-3 sm:max-w-[95vw]">
          <DialogTitle className="truncate pr-8">
            {target.docStem} — page {target.pageNo}
          </DialogTitle>
          <div className="flex-1 overflow-auto">
            {/* eslint-disable-next-line @next/next/no-img-element -- rendered on demand by the backend (pypdfium2), not a static/optimizable asset */}
            <img src={src} alt={alt} className="mx-auto h-auto w-full max-w-3xl rounded-md" />
          </div>
        </DialogContent>
      </Dialog>
    </aside>
  );
}
