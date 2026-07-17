"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getCollection, listCollectionDocuments } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChunkPreviewList } from "@/components/shared/chunk-preview-list";
import { CollectionStatusBadge } from "@/components/collections/collection-status-badge";
import { ArrowLeft, Loader2 } from "lucide-react";

export function CollectionDetail({ name }: { name: string }) {
  const { data: collection, isLoading } = useQuery({
    queryKey: ["collection", name],
    queryFn: () => getCollection(name),
    refetchInterval: (query) => (query.state.data?.status === "ingesting" ? 3000 : false),
  });
  const { data: documents } = useQuery({
    queryKey: ["collection-documents", name],
    queryFn: () => listCollectionDocuments(name),
  });

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-8 py-10">
      <div>
        <Link href="/collections" className="mb-3 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3.5" /> Collections
        </Link>
        <h1 className="text-xl font-semibold tracking-tight">{name}</h1>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading…
        </div>
      )}

      {collection && (
        <Card>
          <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Strategy" value={<Badge variant="outline">{collection.chunking_strategy}</Badge>} />
            <Stat label="Embedding model" value={collection.embedding_model} />
            <Stat label="Dense dim" value={String(collection.dense_dim)} />
            <Stat label="Sparse model" value={collection.sparse_model ?? "—"} />
            <Stat label="Status" value={<CollectionStatusBadge status={collection.status} />} />
            <Stat label="Chunks" value={String(collection.chunk_count)} />
            <Stat label="Documents" value={String(collection.doc_count)} />
            <Stat label="Created" value={new Date(collection.created_at).toLocaleString()} />
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="documents">
        <TabsList>
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="preview">Preview chunks</TabsTrigger>
        </TabsList>
        <TabsContent value="documents">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Format</TableHead>
                <TableHead className="text-right">Pages</TableHead>
                <TableHead>First ingested</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {documents?.map((doc) => (
                <TableRow key={doc.doc_id}>
                  <TableCell className="font-medium">{doc.filename}</TableCell>
                  <TableCell className="text-muted-foreground">{doc.input_format ?? "—"}</TableCell>
                  <TableCell className="text-right">{doc.num_pages ?? "—"}</TableCell>
                  <TableCell className="text-muted-foreground">{new Date(doc.first_ingested_at).toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {documents && documents.length === 0 && <p className="py-4 text-sm text-muted-foreground">No documents.</p>}
        </TabsContent>
        <TabsContent value="preview">
          <ChunkPreviewList collectionName={name} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}
