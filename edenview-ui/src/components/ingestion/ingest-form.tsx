"use client";

import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import Link from "next/link";
import { ingestFile, listChunkingStrategies, listCollections, listDbs } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Combobox } from "@/components/ui/combobox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { FieldHelp } from "@/components/shared/field-help";
import { UploadCloud, Loader2, Info, Database, Table2, X, FileText } from "lucide-react";

const STRATEGY_HELP: Record<string, string> = {
  recursive_overlap:
    "Fixed-size text splitting with overlap between consecutive chunks. Simple and fast, no document-structure awareness — a reasonable default for plain text-heavy documents.",
  hybrid_docling:
    "Token-aware chunking that respects the document's actual structure (headings, sections, tables) and merges undersized adjacent chunks. Better boundaries than fixed-size splitting for structured documents.",
  parent_child:
    "Small \"child\" chunks are embedded and matched precisely, but each one carries a link to a larger \"parent\" chunk — so search finds the precise span while the LLM gets fuller surrounding context.",
  contextual:
    "Same structure-aware chunking as hybrid_docling, plus one extra LLM call per chunk that prepends a short sentence describing where it fits in the document — improves retrieval on chunks that read ambiguously out of context, at the cost of one LLM call per chunk during ingestion.",
};

interface IngestResult {
  file: File;
  ok: boolean;
  error?: string;
}

function fileKey(f: File) {
  return `${f.name}:${f.size}`;
}

export function IngestForm() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [dbName, setDbName] = useState("");
  const [collectionName, setCollectionName] = useState("");
  const [strategy, setStrategy] = useState("recursive_overlap");
  const [includeImageDescriptions, setIncludeImageDescriptions] = useState(false);
  const [forceFullPageOcr, setForceFullPageOcr] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const { data: dbs } = useQuery({ queryKey: ["dbs"], queryFn: listDbs });
  const { data: collections } = useQuery({ queryKey: ["collections", "all"], queryFn: () => listCollections() });
  const { data: strategies } = useQuery({ queryKey: ["strategies"], queryFn: listChunkingStrategies });

  const addFiles = (incoming: FileList | File[]) => {
    setFiles((prev) => {
      const existing = new Set(prev.map(fileKey));
      const toAdd = Array.from(incoming).filter((f) => !existing.has(fileKey(f)));
      return [...prev, ...toAdd];
    });
  };

  const removeFile = (key: string) => setFiles((prev) => prev.filter((f) => fileKey(f) !== key));

  // Every selected file shares the same db/collection/strategy -- each is ingested
  // via its own POST /ingest call (the API is single-file), fired concurrently.
  // POST /ingest itself returns fast (it only resolves/creates the collection and
  // queues a job) and hands the slow extract/chunk/embed work to a FastAPI
  // BackgroundTask, which Starlette runs in its own worker thread -- so N files
  // already extract/embed concurrently with zero backend changes, bounded by that
  // thread pool and, for the CPU-heavy parsing steps, by Python's GIL (real
  // parallelism there depends on how much of Docling's/pypdfium2's C extension code
  // releases it -- I/O-bound steps like the Ollama embedding calls parallelize
  // cleanly regardless).
  const mutation = useMutation({
    mutationFn: async (filesToIngest: File[]): Promise<IngestResult[]> => {
      return Promise.all(
        filesToIngest.map(async (file): Promise<IngestResult> => {
          const form = new FormData();
          form.append("file", file);
          form.append("db_name", dbName.trim());
          form.append("collection_name", collectionName.trim());
          form.append("strategy", strategy);
          form.append("include_image_descriptions", String(includeImageDescriptions));
          form.append("force_full_page_ocr", String(forceFullPageOcr));
          try {
            await ingestFile(form);
            return { file, ok: true };
          } catch (err) {
            return { file, ok: false, error: (err as Error).message };
          }
        }),
      );
    },
    onSuccess: (results) => {
      const succeeded = results.filter((r) => r.ok);
      const failed = results.filter((r) => !r.ok);

      if (succeeded.length > 0) {
        toast.success(`Queued ${succeeded.length} file${succeeded.length > 1 ? "s" : ""} for "${collectionName.trim()}"`);
      }
      for (const r of failed) {
        toast.error(`${r.file.name}: ${r.error}`);
      }

      // The Ingestion page's job list is server-backed (GET /jobs) -- refetch it so
      // the newly queued job(s) show up immediately instead of waiting for its own
      // poll tick.
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["dbs"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setCollectionName("");
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
  });

  const canSubmit = dbName.trim() && collectionName.trim() && strategy && files.length > 0 && !mutation.isPending;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;
    mutation.mutate(files);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex rounded-lg border border-border p-1">
        <Button variant="secondary" size="sm" className="flex-1" type="button">
          <Database className="size-3.5" /> Vector DB
        </Button>
        <Button variant="ghost" size="sm" className="flex-1 opacity-50" type="button" disabled>
          <Table2 className="size-3.5" /> SQL DB
          <span className="ml-1 rounded-full bg-muted px-1.5 py-0.5 text-[10px]">coming soon</span>
        </Button>
      </div>

      <Alert>
        <Info className="size-4" />
        <AlertDescription>
          The embedding model (and its dimension), tokenizer, reranker, and every LLM used here
          (contextual chunking, chat, picture description) are all configured on the{" "}
          <Link href="/settings" className="underline underline-offset-2 hover:text-foreground">
            Settings
          </Link>{" "}
          page, not here.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>New ingestion</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
                dragActive ? "border-primary bg-accent/40" : "border-border hover:bg-muted/40"
              }`}
            >
              <UploadCloud className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                {files.length > 0
                  ? `${files.length} file${files.length > 1 ? "s" : ""} selected — drop more, or click to add`
                  : "Drag files here, or click to browse (multiple allowed)"}
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => e.target.files && addFiles(e.target.files)}
              />
            </div>

            {files.length > 0 && (
              <ul className="flex max-h-40 flex-col gap-1 overflow-y-auto rounded-lg border border-border p-2">
                {files.map((f) => (
                  <li
                    key={fileKey(f)}
                    className="flex items-center gap-2 rounded-md px-2 py-1 text-sm hover:bg-muted/50"
                  >
                    <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate">{f.name}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(fileKey(f));
                      }}
                      className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-destructive"
                    >
                      <X className="size-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <Label htmlFor="db-name">Database</Label>
                  <FieldHelp>
                    A logical grouping label for collections — catalog-only, Qdrant itself has no concept
                    of a "database". Pick an existing name to add to it, or type a new one to create it.
                  </FieldHelp>
                </div>
                <Combobox
                  id="db-name"
                  placeholder="e.g. my-first-db"
                  value={dbName}
                  onValueChange={setDbName}
                  items={dbs?.map((db) => db.name) ?? []}
                  emptyText="No existing databases — press Enter to create this one"
                />
                <p className="text-xs text-muted-foreground">Pick an existing one, or type a new name to create it.</p>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <Label htmlFor="collection-name">Collection name</Label>
                  <FieldHelp>
                    The actual Qdrant collection this document's chunks are written into — one chunking
                    strategy and embedding model per collection. Globally unique across every database.
                    All files selected above are ingested into this same collection, as separate documents.
                  </FieldHelp>
                </div>
                <Combobox
                  id="collection-name"
                  placeholder="e.g. fiscal-health"
                  value={collectionName}
                  onValueChange={setCollectionName}
                  items={collections?.map((c) => c.qdrant_collection_name) ?? []}
                  emptyText="No existing collections — press Enter to create this one"
                />
                <p className="text-xs text-muted-foreground">
                  Pick an existing collection to add to it, or type a new name — must be unique across all databases.
                </p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <Label>Chunking strategy</Label>
                  <FieldHelp>{STRATEGY_HELP[strategy] ?? "Choose how this document gets split into chunks."}</FieldHelp>
                </div>
                <Select value={strategy} onValueChange={(v) => setStrategy(v ?? "recursive_overlap")}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    {strategies?.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-3 pt-6">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="image-descriptions"
                    checked={includeImageDescriptions}
                    onCheckedChange={(v) => setIncludeImageDescriptions(v === true)}
                  />
                  <Label htmlFor="image-descriptions" className="font-normal text-muted-foreground">
                    Generate image descriptions
                  </Label>
                  <FieldHelp>
                    Runs the configured picture-description vision model once per retained image in the
                    document and folds a generated caption into a searchable chunk. Meaningfully slower
                    ingestion (one extra model call per image) — leave this off unless you need images to
                    be searchable by their content. The vision model (Settings → picture description model)
                    must already be pulled in Ollama, or every image description call will fail.
                  </FieldHelp>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="force-full-page-ocr"
                    checked={forceFullPageOcr}
                    onCheckedChange={(v) => setForceFullPageOcr(v === true)}
                  />
                  <Label htmlFor="force-full-page-ocr" className="font-normal text-muted-foreground">
                    Scanned document (force full-page OCR)
                  </Label>
                  <FieldHelp>
                    By default, OCR already runs automatically wherever it's needed — Docling detects
                    which regions of a page are actual scanned images vs. real digital text, and only
                    OCRs the former. Enable this only if you know this document is a scan (or a shaky
                    one) and want every page fully OCR'd regardless of that detection — slower, but a
                    safety net against Docling misjudging a page. Leave off for ordinary digital
                    documents.
                  </FieldHelp>
                </div>
              </div>
            </div>

            <Button type="submit" disabled={!canSubmit} className="w-fit">
              {mutation.isPending && <Loader2 className="size-4 animate-spin" />}
              {files.length > 1 ? `Start ingestion (${files.length} files)` : "Start ingestion"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
