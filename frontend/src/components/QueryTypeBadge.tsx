interface Props {
  queryType: string;
}

const COLORS: Record<string, string> = {
  factual: "bg-blue-100 text-blue-700",
  comparative: "bg-purple-100 text-purple-700",
  multihop: "bg-teal-100 text-teal-700",
};

export default function QueryTypeBadge({ queryType }: Props) {
  const cls = COLORS[queryType] || "bg-gray-100 text-gray-700";
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${cls}`}>
      {queryType}
    </span>
  );
}
