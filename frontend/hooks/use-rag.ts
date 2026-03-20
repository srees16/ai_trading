import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { RAGResponse, RAGSource, RAGChunk } from "@/lib/types";
import { useState, useCallback, useRef } from "react";

export function useRagSources() {
  const qc = useQueryClient();

  const sourcesQuery = useQuery({
    queryKey: ["rag-sources"],
    queryFn: () => api.get<RAGSource[]>("/api/v1/rag/sources"),
  });

  const deleteMutation = useMutation({
    mutationFn: (sourceId: string) =>
      api.del<void>(`/api/v1/rag/sources/${sourceId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-sources"] }),
  });

  const uploadMutation = useMutation({
    mutationFn: (files: File[]) => {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      return api.postForm<{ ingested: number }>("/api/v1/rag/upload", fd);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-sources"] }),
  });

  return {
    sources: sourcesQuery.data ?? [],
    isLoading: sourcesQuery.isLoading,
    refresh: sourcesQuery.refetch,
    deleteSource: deleteMutation.mutate,
    upload: uploadMutation.mutateAsync,
    isUploading: uploadMutation.isPending,
    uploadError: uploadMutation.error?.message ?? null,
  };
}

export function useRagQuery() {
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<RAGChunk[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const query = useCallback(
    (q: string, ragEnabled: boolean, sourceIds?: string[]) => {
      setAnswer("");
      setChunks([]);
      setIsStreaming(true);

      eventSourceRef.current?.close();
      const params: Record<string, string> = {
        q,
        rag: String(ragEnabled),
      };
      if (sourceIds?.length) params.sources = sourceIds.join(",");

      const es = api.createSSE("/api/v1/rag/query", params);
      let buffer = "";

      es.addEventListener("chunk", (e) => {
        try {
          const c = JSON.parse(e.data) as RAGChunk;
          setChunks((prev) => [...prev, c]);
        } catch {}
      });

      es.addEventListener("token", (e) => {
        buffer += e.data;
        setAnswer(buffer);
      });

      es.addEventListener("done", () => {
        setIsStreaming(false);
        es.close();
      });

      es.onerror = () => {
        setIsStreaming(false);
        es.close();
      };

      eventSourceRef.current = es;
    },
    [],
  );

  const cancel = useCallback(() => {
    eventSourceRef.current?.close();
    setIsStreaming(false);
  }, []);

  return { query, answer, chunks, isStreaming, cancel };
}
