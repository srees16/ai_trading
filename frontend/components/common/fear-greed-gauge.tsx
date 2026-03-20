"use client";

import { useFearGreed } from "@/hooks/use-macro";

export function FearGreedGauge() {
  const { data, isLoading } = useFearGreed();

  if (isLoading || !data || data.score == null) return null;

  const score = data.score;
  const label = data.label || "N/A";

  const color =
    score <= 20 ? "#dc2626" :
    score <= 40 ? "#ea580c" :
    score <= 60 ? "#ca8a04" :
    score <= 80 ? "#16a34a" : "#15803d";

  return (
    <div
      className="flex items-center gap-3 px-4 py-2 rounded-lg bg-background border"
      style={{ borderLeftWidth: 4, borderLeftColor: color }}
    >
      <span className="text-xs text-muted-foreground font-semibold">India F&G</span>
      <span className="text-xl font-extrabold" style={{ color }}>{score.toFixed(0)}</span>
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="flex-1 h-2 bg-muted rounded-full min-w-[80px]">
        <div className="h-full rounded-full" style={{ width: `${score}%`, background: color }} />
      </div>
    </div>
  );
}
