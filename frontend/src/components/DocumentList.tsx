import type { Document } from "../types";

interface Props {
  documents: Document[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
  loading: boolean;
}

export default function DocumentList({ documents, selectedId, onSelect, onDelete, loading }: Props) {
  if (loading) {
    return <p className="text-sm text-gray-500 text-center py-4">Loading...</p>;
  }

  if (!documents.length) {
    return (
      <p className="text-sm text-gray-500 text-center py-4">
        No documents yet. Upload one above.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      <button
        onClick={() => onSelect(null)}
        className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
          selectedId === null
            ? "bg-indigo-600 text-white"
            : "text-gray-300 hover:bg-gray-800"
        }`}
      >
        All Documents
      </button>
      {documents.map((doc) => (
        <div
          key={doc.id}
          className={`group flex items-center rounded-lg transition-colors ${
            selectedId === doc.id
              ? "bg-indigo-600 text-white"
              : "text-gray-300 hover:bg-gray-800"
          }`}
        >
          <button
            onClick={() => onSelect(doc.id)}
            className="flex-1 text-left px-3 py-2 min-w-0"
          >
            <p className="text-sm font-medium truncate">{doc.filename}</p>
            <p className="text-xs opacity-60">
              {doc.page_count ? `${doc.page_count} pages` : "Processing..."}
            </p>
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(doc.id);
            }}
            className="px-2 py-1 mr-1 rounded text-gray-500 hover:bg-red-500/20 hover:text-red-400 transition-all"
            title="Delete document"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
