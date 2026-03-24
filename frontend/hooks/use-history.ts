import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  SignalHistoryRow,
  BacktestHistoryRow,
  Market,
} from "@/lib/types";

export function useSignalHistory(market: Market, page = 1, limit = 50) {
  return useQuery({
    queryKey: ["signal-history", market, page, limit],
    queryFn: () =>
      api.get<{ data: SignalHistoryRow[]; total: number }>(
        "/api/v1/history/signals",
        { market, page: String(page), limit: String(limit) },
      ),
  });
}

export function useBacktestHistory(market: Market, page = 1, limit = 50) {
  return useQuery({
    queryKey: ["backtest-history", market, page, limit],
    queryFn: () =>
      api.get<{ data: BacktestHistoryRow[]; total: number }>(
        "/api/v1/history/backtests",
        { market, page: String(page), limit: String(limit) },
      ),
  });
}
