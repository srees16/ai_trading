import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

interface MacroSnapshot {
  vix: number | null;
  vix_label: string;
  index_name: string;
  index_price: number | null;
  index_change_pct: number | null;
  us_10y_yield: number | null;
  gold_price: number | null;
  crude_oil_price: number | null;
  macro_sentiment_label: string | null;
  macro_sentiment_score: number | null;
}

interface FearGreed {
  score: number | null;
  label: string;
}

interface TickerPrice {
  symbol: string;
  price: number;
  change_pct: number;
}

interface TickerPricesResponse {
  is_market_open: boolean;
  prices: TickerPrice[];
}

export function useMacroSnapshot(market: string) {
  return useQuery({
    queryKey: ["macro-snapshot", market],
    queryFn: () => api.get<MacroSnapshot>("/api/v1/macro/snapshot", { market }),
    staleTime: 5 * 60 * 1000, // 5 min
    retry: 1,
  });
}

export function useFearGreed() {
  return useQuery({
    queryKey: ["fear-greed"],
    queryFn: () => api.get<FearGreed>("/api/v1/macro/fear-greed"),
    staleTime: 10 * 60 * 1000, // 10 min
    retry: 1,
  });
}

export function useTickerPrices(symbols: string[], market: string) {
  const query = useQuery({
    queryKey: ["ticker-prices", market, symbols],
    queryFn: () =>
      api.get<TickerPricesResponse>("/api/v1/market/ticker-prices", {
        symbols: symbols.join(","),
        market,
      }),
    enabled: symbols.length > 0,
    refetchInterval: (query) => {
      const isOpen = query.state.data?.is_market_open;
      return isOpen ? 2_000 : 60_000; // 2s live, 60s after hours
    },
    staleTime: 1_500,
    placeholderData: keepPreviousData,
    retry: 1,
  });

  return {
    ...query,
    prices: query.data?.prices ?? [],
    isMarketOpen: query.data?.is_market_open ?? false,
  };
}
