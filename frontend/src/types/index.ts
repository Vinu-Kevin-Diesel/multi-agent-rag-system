export interface Document {
  id: string;
  filename: string;
  content_type: string | null;
  page_count: number | null;
  created_at: string;
}

export interface IngestResponse {
  document_id: string;
  filename: string;
  num_chunks: number;
  page_count: number | null;
}

export interface SourceChunk {
  chunk_id: string;
  content: string;
  page_number: number | null;
  similarity: number;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
  document_id?: string | null;
}

export interface QueryResponse {
  answer: string;
  query_type: string;
  confidence: number;
  sources: SourceChunk[];
  retrieval_attempts: number;
}
