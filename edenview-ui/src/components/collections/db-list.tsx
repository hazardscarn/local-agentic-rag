"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { deleteCollection, deleteDb, listCollections, listDbs } from "@/lib/api";
import { ApiError } from "@/lib/api";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { CreateDbDialog } from "@/components/collections/create-db-dialog";
import { CollectionStatusBadge } from "@/components/collections/collection-status-badge";
import { Loader2, Trash2, Eye, Database } from "lucide-react";

function CollectionsTable({ dbName }: { dbName: string }) {
  const queryClient = useQueryClient();
  const { data: collections, isLoading } = useQuery({
    queryKey: ["collections", dbName],
    queryFn: () => listCollections(dbName),
    // Poll while anything's still ingesting so the status flips to "ready"/"error"
    // on its own instead of requiring a manual refresh.
    refetchInterval: (query) => (query.state.data?.some((c) => c.status === "ingesting") ? 3000 : false),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteCollection,
    onSuccess: (_void, name) => {
      toast.success(`Collection "${name}" deleted`);
      queryClient.invalidateQueries({ queryKey: ["collections", dbName] });
      queryClient.invalidateQueries({ queryKey: ["dbs"] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading collections…
      </div>
    );
  }

  if (!collections || collections.length === 0) {
    return <p className="py-4 text-sm text-muted-foreground">No collections in this database yet.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Collection</TableHead>
          <TableHead>Strategy</TableHead>
          <TableHead>Embedding</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Chunks</TableHead>
          <TableHead className="text-right">Docs</TableHead>
          <TableHead className="w-24" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {collections.map((c) => (
          <TableRow key={c.collection_id}>
            <TableCell className="font-medium">{c.qdrant_collection_name}</TableCell>
            <TableCell>
              <Badge variant="outline">{c.chunking_strategy}</Badge>
            </TableCell>
            <TableCell className="text-muted-foreground">{c.embedding_model}</TableCell>
            <TableCell>
              <CollectionStatusBadge status={c.status} />
            </TableCell>
            <TableCell className="text-right">{c.chunk_count}</TableCell>
            <TableCell className="text-right">{c.doc_count}</TableCell>
            <TableCell>
              <div className="flex justify-end gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  nativeButton={false}
                  render={<Link href={`/collections/${encodeURIComponent(c.qdrant_collection_name)}`} />}
                >
                  <Eye className="size-3.5" />
                </Button>
                <ConfirmDialog
                  trigger={
                    <Button variant="ghost" size="icon" className="size-7 text-destructive hover:text-destructive">
                      <Trash2 className="size-3.5" />
                    </Button>
                  }
                  title={`Delete "${c.qdrant_collection_name}"?`}
                  description="This deletes the Qdrant collection and its catalog rows. Source document images are kept in case another collection references them."
                  onConfirm={() => deleteMutation.mutate(c.qdrant_collection_name)}
                  isPending={deleteMutation.isPending}
                />
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function DbList() {
  const queryClient = useQueryClient();
  const { data: dbs, isLoading } = useQuery({ queryKey: ["dbs"], queryFn: listDbs });

  const deleteDbMutation = useMutation({
    mutationFn: deleteDb,
    onSuccess: () => {
      toast.success("Database deleted");
      queryClient.invalidateQueries({ queryKey: ["dbs"] });
    },
    onError: (err: Error) => {
      if (err instanceof ApiError && err.status === 409) {
        toast.error(err.message || "Delete every collection under this database first.");
      } else {
        toast.error(err.message);
      }
    },
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-muted-foreground">Databases</h2>
        <CreateDbDialog />
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading databases…
        </div>
      )}

      {dbs && dbs.length === 0 && (
        <p className="rounded-lg border border-dashed border-border py-8 text-center text-sm text-muted-foreground">
          No databases yet — ingest a document to create one.
        </p>
      )}

      {dbs && dbs.length > 0 && (
        <Accordion className="flex flex-col gap-3">
          {dbs.map((db) => (
            <AccordionItem key={db.db_id} value={db.db_id} className="rounded-lg border border-border bg-card px-4">
              <div className="flex items-center gap-2">
                <AccordionTrigger className="py-3 hover:no-underline">
                  <span className="flex items-center gap-2 text-sm font-medium">
                    <Database className="size-4 text-muted-foreground" />
                    {db.name}
                  </span>
                </AccordionTrigger>
                <ConfirmDialog
                  trigger={
                    <Button variant="ghost" size="icon" className="size-7 shrink-0 text-destructive hover:text-destructive">
                      <Trash2 className="size-3.5" />
                    </Button>
                  }
                  title={`Delete database "${db.name}"?`}
                  description="Only succeeds once every collection under this database is gone."
                  onConfirm={() => deleteDbMutation.mutate(db.db_id)}
                  isPending={deleteDbMutation.isPending}
                />
              </div>
              <AccordionContent>
                <CollectionsTable dbName={db.name} />
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      )}
    </div>
  );
}
