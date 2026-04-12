import ReactMarkdown from "react-markdown";
import type { QueryResponse } from "../types";
import ConfidenceBadge from "./ConfidenceBadge";
import QueryTypeBadge from "./QueryTypeBadge";
import SourceList from "./SourceList";

interface Props {
  result: QueryResponse;
}

export default function AnswerDisplay({ result }: Props) {
  return (
    <div className="space-y-6">
      {/* Header badges */}
      <div className="flex items-center gap-3 flex-wrap">
        <QueryTypeBadge queryType={result.query_type} />
        <ConfidenceBadge confidence={result.confidence} />
        {result.retrieval_attempts > 1 && (
          <span className="text-xs text-gray-400">
            {result.retrieval_attempts} retrieval attempts
          </span>
        )}
      </div>

      {/* Answer */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
        <div className="prose prose-indigo max-w-none prose-sm">
          <ReactMarkdown>{result.answer}</ReactMarkdown>
        </div>
      </div>

      {/* Sources */}
      <SourceList sources={result.sources} />
    </div>
  );
}
