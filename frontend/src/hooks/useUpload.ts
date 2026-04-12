import { useState } from "react";
import { uploadDocument } from "../api/client";
import type { IngestResponse } from "../types";

export function useUpload(onSuccess?: (res: IngestResponse) => void) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = async (file: File) => {
    try {
      setUploading(true);
      setError(null);
      const result = await uploadDocument(file);
      onSuccess?.(result);
      return result;
    } catch (err: any) {
      setError(err.message || "Upload failed");
      return null;
    } finally {
      setUploading(false);
    }
  };

  return { upload, uploading, error };
}
