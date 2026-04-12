import { useState } from "react";
import { useDocuments } from "./hooks/useDocuments";
import { useUpload } from "./hooks/useUpload";
import { useQuery } from "./hooks/useQuery";
import { deleteDocument } from "./api/client";
import DocumentUpload from "./components/DocumentUpload";
import DocumentList from "./components/DocumentList";
import QueryInput from "./components/QueryInput";
import AnswerDisplay from "./components/AnswerDisplay";

export default function App() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const { documents, loading: docsLoading, refresh } = useDocuments();
  const { upload, uploading, error: uploadError } = useUpload(() => refresh());
  const { result, loading: querying, error: queryError, query } = useQuery();

  const handleDelete = async (id: string) => {
    try {
      await deleteDocument(id);
      if (selectedDocId === id) setSelectedDocId(null);
      refresh();
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const handleQuery = (question: string) => {
    query({
      question,
      top_k: selectedDocId ? 5 : 10 * documents.length,
      document_id: selectedDocId,
    });
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-80 bg-gray-900 text-white flex flex-col shrink-0">
        <div className="p-5 border-b border-gray-800">
          <h1 className="text-lg font-bold flex items-center gap-2">
            <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            DocIntel Agent
          </h1>
          <p className="text-xs text-gray-500 mt-1">Multi-agent RAG System</p>
        </div>

        <div className="p-4 space-y-4 flex-1 overflow-y-auto">
          <div>
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Upload Document
            </h2>
            <DocumentUpload onUpload={upload} uploading={uploading} />
            {uploadError && (
              <p className="text-xs text-red-400 mt-2">{uploadError}</p>
            )}
          </div>

          <div>
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Documents
            </h2>
            <DocumentList
              documents={documents}
              selectedId={selectedDocId}
              onSelect={setSelectedDocId}
              onDelete={handleDelete}
              loading={docsLoading}
            />
          </div>
        </div>

        <div className="p-4 border-t border-gray-800 text-xs text-gray-600">
          Powered by Kimi K2.5 + pgvector
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Query input */}
            <div>
              <h2 className="text-xl font-bold text-gray-800 mb-1">Ask your documents</h2>
              <p className="text-sm text-gray-500 mb-4">
                {selectedDocId
                  ? `Querying: ${documents.find((d) => d.id === selectedDocId)?.filename || "selected document"}`
                  : "Searching across all documents"}
              </p>
              <QueryInput onSubmit={handleQuery} loading={querying} />
              {queryError && (
                <p className="text-sm text-red-500 mt-2">{queryError}</p>
              )}
            </div>

            {/* Results */}
            {querying && !result && (
              <div className="flex items-center justify-center py-16">
                <div className="flex flex-col items-center gap-3">
                  <svg className="animate-spin h-8 w-8 text-indigo-500" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <p className="text-sm text-gray-500">Routing query through agents...</p>
                </div>
              </div>
            )}

            {result && <AnswerDisplay result={result} />}

            {/* Empty state */}
            {!result && !querying && (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <svg className="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <p className="text-lg font-medium">No queries yet</p>
                <p className="text-sm mt-1">Upload a document and ask a question to get started</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
