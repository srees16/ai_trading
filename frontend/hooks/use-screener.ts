import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  ScreenerConfig,
  RiskConfig,
  ScreenedStock,
  TradePlan,
  OrderResult,
  TradeMonitorSummary,
} from "@/lib/types";
import { useQuery } from "@tanstack/react-query";

interface ScreenerResponse {
  stocks: ScreenedStock[];
  trade_plans: TradePlan[];
  summary: Record<string, unknown>;
}

export function useScreener() {
  const qc = useQueryClient();

  const screenMutation = useMutation({
    mutationFn: (params: { screener: ScreenerConfig; risk: RiskConfig; tickers: string[] }) =>
      api.post<ScreenerResponse>("/api/v1/screener/run", params),
  });

  const executeMutation = useMutation({
    mutationFn: (plans: TradePlan[]) =>
      api.post<OrderResult[]>("/api/v1/screener/execute", { plans }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trade-monitor"] });
    },
  });

  const monitorQuery = useQuery({
    queryKey: ["trade-monitor"],
    queryFn: () => api.get<TradeMonitorSummary>("/api/v1/screener/monitor"),
    refetchInterval: 30000,
    enabled: false,
  });

  return {
    screen: screenMutation.mutate,
    screenAsync: screenMutation.mutateAsync,
    isScreening: screenMutation.isPending,
    screenResult: screenMutation.data,
    screenError: screenMutation.error?.message ?? null,
    execute: executeMutation.mutate,
    isExecuting: executeMutation.isPending,
    executeResults: executeMutation.data,
    executeError: executeMutation.error?.message ?? null,
    monitor: monitorQuery.data,
    refreshMonitor: monitorQuery.refetch,
  };
}
