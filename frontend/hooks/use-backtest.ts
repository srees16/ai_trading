import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  BacktestRequest,
  BacktestResult,
  StrategyInfo,
  Market,
} from "@/lib/types";

export function useBacktest(market: Market) {
  const qc = useQueryClient();

  const strategiesQuery = useQuery({
    queryKey: ["strategies", market],
    queryFn: () =>
      api.get<StrategyInfo[]>("/api/v1/backtest/strategies", { market }),
  });

  const runMutation = useMutation({
    mutationFn: (req: BacktestRequest) =>
      api.post<BacktestResult>("/api/v1/backtest/run", req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backtest-history", market] });
    },
  });

  return {
    strategies: strategiesQuery.data ?? [],
    strategiesLoading: strategiesQuery.isLoading,
    run: runMutation.mutate,
    runAsync: runMutation.mutateAsync,
    isRunning: runMutation.isPending,
    result: runMutation.data,
    error: runMutation.error?.message ?? null,
  };
}
