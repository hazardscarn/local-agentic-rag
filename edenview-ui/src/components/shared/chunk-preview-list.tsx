"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { previewCollection, fileUrl } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { PreviewChunk } from "@/lib/types";
import { Loader2, ImageIcon, ChevronDown, ChevronUp } from "lucide-react";

function PreviewChunkCard({ chunk }: { chunk: PreviewChunk }) {
  const [parentOpen, setParentOpen] = useState(false);

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        {chunk.page_no != null && <Badge variant="secondary">page {chunk.page_no}</Badge>}
        <Badge variant="outline">{chunk.kind}</Badge>
        <Badge variant="outline">{chunk.strategy}</Badge>
      </div>

      {chunk.parent_text ? (
        <>
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Matched span (child)
          </p>
          <p className="whitespace-pre-wrap rounded-md bg-accent/40 p-2 text-sm text-foreground/90">{chunk.text}</p>
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 h-auto px-0 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setParentOpen((v) => !v)}
          >
            {parentOpen ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
            {parentOpen ? "Hide" : "Show"} full parent context
          </Button>
          {parentOpen && (
            <div className="mt-1.5">
              <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Full parent context (fed to the LLM at query time)
              </p>
              <p className="whitespace-pre-wrap text-sm text-foreground/80">{chunk.parent_text}</p>
            </div>
          )}
        </>
      ) : (
        <p className="whitespace-pre-wrap text-sm text-foreground/90 line-clamp-6">{chunk.text}</p>
      )}

      {chunk.images.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {chunk.images.map((img) => (
            <a
              key={img.picture_id}
              href={fileUrl(img.image_path)}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 rounded-md border border-border bg-muted px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              <ImageIcon className="size-3" />
              {img.kind}
              {img.page_no != null ? ` · p.${img.page_no}` : ""}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export function ChunkPreviewList({ collectionName }: { collectionName: string }) {
  const [offsetStack, setOffsetStack] = useState<(string | null)[]>([null]);
  const offset = offsetStack[offsetStack.length - 1];

  const { data, isLoading, isError } = useQuery({
    queryKey: ["preview", collectionName, offset],
    queryFn: () => previewCollection(collectionName, 10, offset),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading chunks…
      </div>
    );
  }
  if (isError || !data) {
    return <p className="py-4 text-sm text-destructive">Failed to load chunk preview.</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      {data.chunks.length === 0 && <p className="py-4 text-sm text-muted-foreground">No chunks found.</p>}
      {data.chunks.map((chunk) => (
        <PreviewChunkCard key={chunk.chunk_id} chunk={chunk} />
      ))}

      <div className="flex items-center justify-between pt-1">
        <Button
          variant="outline"
          size="sm"
          disabled={offsetStack.length <= 1}
          onClick={() => setOffsetStack((s) => s.slice(0, -1))}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!data.next_offset}
          onClick={() => setOffsetStack((s) => [...s, data.next_offset])}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
