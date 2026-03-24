import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  AnalysisResponse,
  Market,
} from "@/lib/types";

interface RunAnalysisParams {
  tickers: string[];
  market: Market;
  period?: string;
}

export function useAnalysis(market: Market) {
  const qc = useQueryClient();

  const runMutation = useMutation({
    mutationFn: (params: RunAnalysisParams) =>
      api.post<AnalysisResponse>("/api/v1/analysis/run", params),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["analysis", market] });
      qc.invalidateQueries({ queryKey: ["history", market] });
    },
  });

  const latestQuery = useQuery({
    queryKey: ["analysis", market, "latest"],
    queryFn: () =>
      api.get<AnalysisResponse>("/api/v1/analysis/latest", { market }),
    enabled: false,
  });

  return {
    run: runMutation.mutate,
    runAsync: runMutation.mutateAsync,
    isRunning: runMutation.isPending,
    result: runMutation.data ?? latestQuery.data,
    error: runMutation.error?.message ?? latestQuery.error?.message ?? null,
    fetchLatest: latestQuery.refetch,
  };
}
