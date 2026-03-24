import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { IndexQuote, OptionChainRow } from "@/lib/types";

export function useIndexQuotes() {
  return useQuery({
    queryKey: ["index-quotes"],
    queryFn: () => api.get<IndexQuote[]>("/api/v1/options/indices"),
    refetchInterval: 10000,
  });
}

export function useOptionChain(symbol: string, expiry: string) {
  return useQuery({
    queryKey: ["option-chain", symbol, expiry],
    queryFn: () =>
      api.get<OptionChainRow[]>("/api/v1/options/chain", { symbol, expiry }),
    enabled: !!symbol && !!expiry,
  });
}

export function useOptionExpiries(symbol: string) {
  return useQuery({
    queryKey: ["option-expiries", symbol],
    queryFn: () =>
      api.get<string[]>("/api/v1/options/expiries", { symbol }),
    enabled: !!symbol,
  });
}
