import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { DWCredentials, DWAccount, DWPosition } from "@/lib/types";

const DW_TOKEN_KEY = "dw_token";
const DW_TOKEN_TS_KEY = "dw_token_ts";
const DW_TOKEN_TTL_MS = 8 * 60 * 60 * 1000; // 8 hours

function isDWTokenExpired(): boolean {
  const ts = localStorage.getItem(DW_TOKEN_TS_KEY);
  if (!ts) return true;
  return Date.now() - Number(ts) > DW_TOKEN_TTL_MS;
}

export function useDriveWealth() {
  const qc = useQueryClient();

  const loginMutation = useMutation({
    mutationFn: (creds: DWCredentials) =>
      api.post<{ token: string; account: DWAccount }>("/api/v1/drivewealth/login", creds),
    onSuccess: (data) => {
      if (data.token) {
        localStorage.setItem(DW_TOKEN_KEY, data.token);
        localStorage.setItem(DW_TOKEN_TS_KEY, String(Date.now()));
      }
      qc.invalidateQueries({ queryKey: ["dw-account"] });
    },
  });

  const accountQuery = useQuery({
    queryKey: ["dw-account"],
    queryFn: () => api.get<DWAccount>("/api/v1/drivewealth/account"),
    enabled: false,
  });

  const positionsQuery = useQuery({
    queryKey: ["dw-positions"],
    queryFn: () => api.get<DWPosition[]>("/api/v1/drivewealth/positions"),
    enabled: false,
  });

  const orderMutation = useMutation({
    mutationFn: (order: {
      symbol: string;
      side: "buy" | "sell";
      order_type: "market" | "limit";
      quantity: number;
      limit_price?: number;
    }) => api.post<{ order_id: string; status: string }>("/api/v1/drivewealth/orders", order),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dw-positions"] });
      qc.invalidateQueries({ queryKey: ["dw-account"] });
    },
  });

  return {
    login: loginMutation.mutate,
    isLoggingIn: loginMutation.isPending,
    loginError: loginMutation.error?.message ?? null,
    account: loginMutation.data?.account ?? accountQuery.data ?? null,
    isConnected: !!(loginMutation.data?.account || accountQuery.data) && !isDWTokenExpired(),
    fetchAccount: accountQuery.refetch,
    positions: positionsQuery.data ?? [],
    fetchPositions: positionsQuery.refetch,
    placeOrder: orderMutation.mutate,
    isOrdering: orderMutation.isPending,
    orderError: orderMutation.error?.message ?? null,
    logout: () => {
      localStorage.removeItem(DW_TOKEN_KEY);
      localStorage.removeItem(DW_TOKEN_TS_KEY);
      qc.removeQueries({ queryKey: ["dw-account"] });
      qc.removeQueries({ queryKey: ["dw-positions"] });
      loginMutation.reset();
    },
  };
}
