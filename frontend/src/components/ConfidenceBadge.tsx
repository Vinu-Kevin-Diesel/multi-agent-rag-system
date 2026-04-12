interface Props {
  confidence: number;
}

export default function ConfidenceBadge({ confidence }: Props) {
  const pct = Math.round(confidence * 100);
  const color =
    confidence >= 0.8
      ? "bg-green-500"
      : confidence >= 0.5
        ? "bg-yellow-500"
        : "bg-red-500";
  const textColor =
    confidence >= 0.8
      ? "text-green-700"
      : confidence >= 0.5
        ? "text-yellow-700"
        : "text-red-700";
  const bgLight =
    confidence >= 0.8
      ? "bg-green-50"
      : confidence >= 0.5
        ? "bg-yellow-50"
        : "bg-red-50";

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${bgLight}`}>
      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-semibold ${textColor}`}>{pct}%</span>
    </div>
  );
}
