import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { RAGResponse, RAGSource, RAGChunk } from "@/lib/types";
import { useState, useCallback, useRef, useEffect } from "react";

interface IngestTask {
  task_id: string;
  file_name: string;
  status: string;
  stage: string;
  stage_pct: number;
  error: string | null;
}

const BACKEND_URL = typeof window !== "undefined"
  ? (process.env.NEXT_PUBLIC_API_URL || "http://localhost:9001")
  : "http://localhost:9001";

function backendHeaders(): HeadersInit {
  const headers: HeadersInit = {};
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export function useRagSources() {
  const qc = useQueryClient();
  const [ingesting, setIngesting] = useState(false);

  const sourcesQuery = useQuery({
    queryKey: ["rag-sources"],
    queryFn: () => api.get<RAGSource[]>("/api/v1/rag/sources"),
  });

  const deleteMutation = useMutation({
    mutationFn: (sourceId: string) =>
      api.del<void>(`/api/v1/rag/sources/${sourceId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-sources"] }),
  });

  // Poll ingest status while ingesting
  const statusQuery = useQuery<IngestTask[]>({
    queryKey: ["rag-ingest-status"],
    queryFn: async () => {
      const res = await fetch(`${BACKEND_URL}/api/v1/rag/ingest-status`, { headers: backendHeaders() });
      return res.json();
    },
    enabled: ingesting,
    refetchInterval: 2000,
  });

  // Auto-stop polling when all tasks are done, and refresh sources
  useEffect(() => {
    if (!ingesting || !statusQuery.data) return;
    const active = statusQuery.data.filter((t) => t.status === "pending" || t.status === "running");
    if (active.length === 0 && statusQuery.data.length > 0) {
      setIngesting(false);
      qc.invalidateQueries({ queryKey: ["rag-sources"] });
    }
  }, [statusQuery.data, ingesting, qc]);

  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      const res = await fetch(`${BACKEND_URL}/api/v1/rag/upload`, {
        method: "POST",
        headers: backendHeaders(),
        body: fd,
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `Upload failed (HTTP ${res.status})`);
      }
      return res.json() as Promise<{ submitted: number; tasks: { task_id: string; file_name: string; status: string }[] }>;
    },
    onSuccess: () => {
      setIngesting(true);
    },
  });

  return {
    sources: sourcesQuery.data ?? [],
    isLoading: sourcesQuery.isLoading,
    refresh: sourcesQuery.refetch,
    deleteSource: deleteMutation.mutate,
    upload: uploadMutation.mutateAsync,
    isUploading: uploadMutation.isPending,
    uploadError: uploadMutation.error?.message ?? null,
    ingestTasks: statusQuery.data ?? [],
    isIngesting: ingesting,
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
        try {
          buffer += JSON.parse(e.data);
        } catch {
          buffer += e.data;
        }
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
