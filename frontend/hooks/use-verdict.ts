import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { VerdictRequest, VerdictResult, Market } from "@/lib/types";

export function useVerdict(market: Market) {
  const qc = useQueryClient();

  const runMutation = useMutation({
    mutationFn: (req: VerdictRequest) =>
      api.postDirect<VerdictResult[]>("/api/v1/verdict/run", req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["history", market] });
    },
  });

  return {
    run: runMutation.mutate,
    runAsync: runMutation.mutateAsync,
    isRunning: runMutation.isPending,
    results: runMutation.data ?? [],
    error: runMutation.error?.message ?? null,
    reset: runMutation.reset,
  };
}
