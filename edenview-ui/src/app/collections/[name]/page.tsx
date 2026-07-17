import { CollectionDetail } from "@/components/collections/collection-detail";

export default async function CollectionDetailPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  return <CollectionDetail name={decodeURIComponent(name)} />;
}
