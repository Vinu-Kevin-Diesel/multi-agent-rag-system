import { useState } from "react";
import type { SourceChunk } from "../types";

interface Props {
  source: SourceChunk;
  index: number;
}

export default function SourceCard({ source, index }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(source.similarity * 100);
  const color =
    source.similarity >= 0.7
      ? "text-green-600"
      : source.similarity >= 0.4
        ? "text-yellow-600"
        : "text-red-600";

  return (
    <div
      className="bg-white border border-gray-200 rounded-lg overflow-hidden cursor-pointer hover:border-indigo-300 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="flex items-center justify-center w-6 h-6 rounded-full bg-indigo-100 text-indigo-600 text-xs font-bold">
            {index + 1}
          </span>
          {source.page_number && (
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
              Page {source.page_number}
            </span>
          )}
          <span className={`text-xs font-semibold ${color}`}>{pct}% match</span>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>
      {!expanded && (
        <p className="px-4 pb-3 text-sm text-gray-500 line-clamp-2">
          {source.content}
        </p>
      )}
      {expanded && (
        <div className="px-4 pb-4 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap border-t border-gray-100 pt-3">
          {source.content}
        </div>
      )}
    </div>
  );
}
