import { DbList } from "@/components/collections/db-list";

export default function CollectionsPage() {
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-8 py-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Collections</h1>
        <p className="text-sm text-muted-foreground">
          The DuckDB-backed catalog: every database, its collections, and what's inside them.
        </p>
      </div>
      <DbList />
    </div>
  );
}
