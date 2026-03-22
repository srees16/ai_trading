"use client";

import { cn, formatNumber, formatPct } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: number;
  deltaLabel?: string;
  formatAsPct?: boolean;
  color?: string;
  className?: string;
}

export function MetricCard({ label, value, delta, deltaLabel, formatAsPct, color, className }: MetricCardProps) {
  const deltaColor = delta === undefined ? "" : delta > 0 ? "pnl-positive" : delta < 0 ? "pnl-negative" : "text-muted-foreground";
  const DeltaIcon = delta === undefined ? null : delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus;

  return (
    <div className={cn("metric-card", className)}>
      <p className="metric-label">{label}</p>
      <p className={cn("metric-value", color)}>
        {typeof value === "number" ? (formatAsPct ? formatPct(value) : formatNumber(value)) : value}
      </p>
      {delta !== undefined && (
        <p className={cn("metric-delta flex items-center gap-1", deltaColor)}>
          {DeltaIcon && <DeltaIcon className="h-3 w-3" />}
          <span>{formatAsPct ? formatPct(delta) : formatNumber(delta)}</span>
          {deltaLabel && <span className="text-muted-foreground text-xs">({deltaLabel})</span>}
        </p>
      )}
    </div>
  );
}

interface MetricsGridProps {
  metrics?: MetricCardProps[];
  columns?: number;
  children?: React.ReactNode;
}

export function MetricsGrid({ metrics, columns = 4, children }: MetricsGridProps) {
  return (
    <div className={cn("grid gap-3", {
      "grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6": columns === 4,
      "grid-cols-1 sm:grid-cols-2 md:grid-cols-3": columns === 3,
      "grid-cols-1 sm:grid-cols-2": columns === 2,
    })}>
      {metrics
        ? metrics.map((m, i) => <MetricCard key={i} {...m} />)
        : children}
    </div>
  );
}
