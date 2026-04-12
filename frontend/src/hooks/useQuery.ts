import { useState } from "react";
import { submitQuery } from "../api/client";
import type { QueryRequest, QueryResponse } from "../types";

export function useQuery() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = async (payload: QueryRequest) => {
    try {
      setLoading(true);
      setError(null);
      const res = await submitQuery(payload);
      setResult(res);
      return res;
    } catch (err: any) {
      setError(err.message || "Query failed");
      return null;
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setResult(null);
    setError(null);
  };

  return { result, loading, error, query, clear };
}
