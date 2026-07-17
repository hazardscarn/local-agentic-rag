"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listJobs, type JobStatusFilter } from "@/lib/api";
import { IngestForm } from "@/components/ingestion/ingest-form";
import { JobRow } from "@/components/ingestion/job-row";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronDown, ChevronUp, Loader2, Search } from "lucide-react";

// Default view is a short, recent-activity feed, not a full history -- searching by
// filename (server-side, not just within these) or picking a bigger limit is how you
// find an older job instead of a page/offset UI.
const DEFAULT_JOBS_LIMIT = 10;
const SEARCH_JOBS_LIMIT = 100;
const LIMIT_OPTIONS = [10, 20, 50, 100];

type StatusFilterOption = "all" | JobStatusFilter;

const STATUS_FILTERS: { value: StatusFilterOption; label: string }[] = [
  { value: "all", label: "All" },
  { value: "active", label: "Active" },
  { value: "done", label: "Done" },
  { value: "error", label: "Error" },
];

export default function IngestionPage() {
  const [expanded, setExpanded] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilterOption>("all");
  const [limit, setLimit] = useState(DEFAULT_JOBS_LIMIT);

  const { data: jobs, isLoading } = useQuery({
    queryKey: ["jobs", search, statusFilter, limit],
    queryFn: () => listJobs(search ? SEARCH_JOBS_LIMIT : limit, search || undefined, statusFilter === "all" ? undefined : statusFilter),
    refetchInterval: (query) => (query.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 2000 : false),
  });

  const heading = search ? "Search results" : statusFilter === "all" ? "Recent jobs" : `${STATUS_FILTERS.find((f) => f.value === statusFilter)?.label} jobs`;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-8 py-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Ingestion</h1>
        <p className="text-sm text-muted-foreground">Upload a document, pick a chunking strategy, and track it through extraction, chunking, and embedding.</p>
      </div>

      <IngestForm />

      <div className="flex flex-col gap-3">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-1.5 text-sm font-semibold text-muted-foreground hover:text-foreground"
        >
          {expanded ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
          {heading}
          {jobs && jobs.length > 0 ? ` (${jobs.length})` : ""}
        </button>

        {expanded && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex gap-1 rounded-lg border border-border p-1">
              {STATUS_FILTERS.map((f) => (
                <Button
                  key={f.value}
                  type="button"
                  size="sm"
                  variant={statusFilter === f.value ? "secondary" : "ghost"}
                  className="h-6 px-2 text-xs"
                  onClick={() => setStatusFilter(f.value)}
                >
                  {f.label}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              {!search && (
                <Select value={String(limit)} onValueChange={(v) => setLimit(Number(v ?? DEFAULT_JOBS_LIMIT))}>
                  <SelectTrigger className="h-7 w-24 text-xs">
                    <SelectValue>{(v: string | null) => `Show ${v ?? limit}`}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {LIMIT_OPTIONS.map((n) => (
                      <SelectItem key={n} value={String(n)}>
                        Show {n}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              <div className="relative w-48">
                <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by filename…"
                  className="h-7 pl-7 text-xs"
                />
              </div>
            </div>
          </div>
        )}

        {expanded && (
          <>
            {isLoading && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" /> Loading…
              </div>
            )}
            {jobs?.length === 0 && (
              <p className="text-sm text-muted-foreground">
                {search ? `No jobs matching "${search}".` : "No ingestion jobs match this filter."}
              </p>
            )}
            {jobs && jobs.length > 0 && (
              <div className="flex max-h-[32rem] flex-col gap-3 overflow-y-auto rounded-lg border border-border p-3">
                {jobs.map((job) => (
                  <JobRow key={job.job_id} job={job} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
