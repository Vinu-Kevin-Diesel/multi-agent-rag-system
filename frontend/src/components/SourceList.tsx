import type { SourceChunk } from "../types";
import SourceCard from "./SourceCard";

interface Props {
  sources: SourceChunk[];
}

export default function SourceList({ sources }: Props) {
  if (!sources.length) return null;

  const sorted = [...sources].sort((a, b) => b.similarity - a.similarity);

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
        Sources ({sources.length} chunks)
      </h3>
      <div className="space-y-2">
        {sorted.map((s, i) => (
          <SourceCard key={s.chunk_id} source={s} index={i} />
        ))}
      </div>
    </div>
  );
}
