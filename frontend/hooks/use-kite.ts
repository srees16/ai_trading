import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  LiveQuote,
  KiteHolding,
  KitePosition,
  KiteOrder,
} from "@/lib/types";

interface KiteSessionStatus {
  active: boolean;
  profile: Record<string, unknown> | null;
}

export function useKiteSessionStatus() {
  return useQuery({
    queryKey: ["kite-session-status"],
    queryFn: () => api.get<KiteSessionStatus>("/api/v1/kite/session/status"),
    refetchInterval: 30000,
  });
}

export function useKiteSessionStart() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<{
        success: boolean;
        profile?: Record<string, unknown>;
        message?: string;
        needs_login?: boolean;
        login_url?: string;
      }>("/api/v1/kite/session/start"),
    onSuccess: (data) => {
      if (data.success) {
        qc.invalidateQueries({ queryKey: ["kite-session-status"] });
        qc.invalidateQueries({ queryKey: ["kite-quotes"] });
        qc.invalidateQueries({ queryKey: ["kite-holdings"] });
        qc.invalidateQueries({ queryKey: ["kite-positions"] });
        qc.invalidateQueries({ queryKey: ["kite-orders"] });
      }
    },
  });
}

export function useKiteSessionComplete() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (requestToken: string) =>
      api.post<{ success: boolean; profile: Record<string, unknown> }>(
        "/api/v1/kite/session/complete",
        { request_token: requestToken },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kite-session-status"] });
      qc.invalidateQueries({ queryKey: ["kite-quotes"] });
      qc.invalidateQueries({ queryKey: ["kite-holdings"] });
      qc.invalidateQueries({ queryKey: ["kite-positions"] });
      qc.invalidateQueries({ queryKey: ["kite-orders"] });
    },
  });
}

export function useKiteSessionStop() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<{ success: boolean }>("/api/v1/kite/session/stop"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kite-session-status"] });
    },
  });
}

export function useKiteQuotes(symbols: string[], enabled = true) {
  return useQuery({
    queryKey: ["kite-quotes", symbols],
    queryFn: () =>
      api.get<LiveQuote[]>("/api/v1/kite/quotes", {
        symbols: symbols.join(","),
      }),
    enabled: enabled && symbols.length > 0,
    refetchInterval: 5000,
  });
}

export function useKiteHoldings() {
  return useQuery({
    queryKey: ["kite-holdings"],
    queryFn: () => api.get<KiteHolding[]>("/api/v1/kite/holdings"),
  });
}

export function useKitePositions() {
  return useQuery({
    queryKey: ["kite-positions"],
    queryFn: () => api.get<KitePosition[]>("/api/v1/kite/positions"),
  });
}

export function useKiteOrders() {
  return useQuery({
    queryKey: ["kite-orders"],
    queryFn: () => api.get<KiteOrder[]>("/api/v1/kite/orders"),
  });
}

export function useKitePlaceOrder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (order: {
      tradingsymbol: string;
      exchange: string;
      transaction_type: "BUY" | "SELL";
      order_type: "MARKET" | "LIMIT" | "SL" | "SL-M";
      quantity: number;
      price?: number;
      trigger_price?: number;
      product?: string;
      validity?: string;
      variety?: string;
    }) => api.post<{ order_id: string }>("/api/v1/kite/orders", order),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kite-orders"] });
      qc.invalidateQueries({ queryKey: ["kite-positions"] });
    },
  });
}
