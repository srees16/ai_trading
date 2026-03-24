import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { ChapterInfo, ChapterResult, AsyncBatchProgress, BatchRunHistoryRow } from "@/lib/types";
import { useState, useCallback, useEffect, useRef } from "react";

export function useTtsChapters() {
  return useQuery({
    queryKey: ["tts-chapters"],
    queryFn: () => api.get<ChapterInfo[]>("/api/v1/tts/chapters"),
  });
}

export interface TtsRunParams {
  chapters: string[];
  tickers?: string[];
  date_start?: string;
  date_end?: string;
}

export function useTtsRun() {
  const [progress, setProgress] = useState<AsyncBatchProgress | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const runMutation = useMutation({
    mutationFn: (params: TtsRunParams) =>
      api.post<{ batch_id: string }>("/api/v1/tts/run", params),
  });

  const startSSE = useCallback(
    (batchId: string) => {
      eventSourceRef.current?.close();
      const es = api.createSSE(`/api/v1/tts/progress/${batchId}`);
      es.onmessage = (e) => {
        try {
          setProgress(JSON.parse(e.data));
        } catch {}
      };
      es.onerror = () => es.close();
      eventSourceRef.current = es;
    },
    [],
  );

  useEffect(() => {
    if (runMutation.data?.batch_id) {
      startSSE(runMutation.data.batch_id);
    }
    return () => eventSourceRef.current?.close();
  }, [runMutation.data?.batch_id, startSSE]);

  const abort = useCallback(async () => {
    const batchId = runMutation.data?.batch_id;
    if (!batchId) return;
    await api.post(`/api/v1/tts/abort/${batchId}`);
    eventSourceRef.current?.close();
  }, [runMutation.data?.batch_id]);

  const isAborted = progress?.status === "aborted";

  return {
    run: runMutation.mutate,
    isRunning: !isAborted && (runMutation.isPending || (progress !== null && progress.completed < progress.total)),
    batchId: runMutation.data?.batch_id ?? null,
    progress,
    error: runMutation.error?.message ?? null,
    abort,
    isAborted,
  };
}

export function useTtsHistory(page = 1, limit = 50) {
  return useQuery({
    queryKey: ["tts-history", page, limit],
    queryFn: () =>
      api.get<{ data: BatchRunHistoryRow[]; total: number }>(
        "/api/v1/tts/history",
        { page: String(page), limit: String(limit) },
      ),
  });
}
