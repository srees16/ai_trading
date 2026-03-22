"use client";

import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "./data-table";
import { DecisionBadge } from "@/components/common/decision-badge";
import { formatNumber } from "@/lib/utils";
import { SENTIMENT_COLORS } from "@/lib/constants";
import type { TradingSignal } from "@/lib/types";

const columns: ColumnDef<TradingSignal, unknown>[] = [
  { accessorFn: (r) => r.news_item.ticker, id: "ticker", header: "Ticker", size: 80 },
  {
    accessorKey: "decision",
    header: "Decision",
    cell: ({ row }) => <DecisionBadge decision={row.original.decision} />,
  },
  {
    accessorKey: "decision_score",
    header: "Score",
    cell: ({ row }) => formatNumber(row.original.decision_score, 3),
  },
  {
    accessorFn: (r) => r.news_item.sentiment_label,
    id: "sentiment",
    header: "Sentiment",
    cell: ({ row }) => {
      const label = row.original.news_item.sentiment_label;
      return (
        <span
          className="px-1.5 py-0.5 rounded text-xs font-medium"
          style={{ color: SENTIMENT_COLORS[label] || "#999" }}
        >
          {label}
        </span>
      );
    },
  },
  {
    accessorFn: (r) => r.news_item.sentiment_confidence,
    id: "confidence",
    header: "Confidence",
    cell: ({ row }) => `${(row.original.news_item.sentiment_confidence * 100).toFixed(1)}%`,
  },
  {
    accessorFn: (r) => r.metrics?.current_price,
    id: "price",
    header: "Price",
    cell: ({ row }) => row.original.metrics ? `$${formatNumber(row.original.metrics.current_price)}` : "—",
  },
  {
    accessorFn: (r) => r.metrics?.rsi,
    id: "rsi",
    header: "RSI",
    cell: ({ row }) => row.original.metrics ? formatNumber(row.original.metrics.rsi, 1) : "—",
  },
  {
    accessorFn: (r) => r.metrics?.altman_z_score,
    id: "z_score",
    header: "Z-Score",
    cell: ({ row }) => row.original.metrics ? formatNumber(row.original.metrics.altman_z_score, 2) : "—",
  },
  {
    accessorFn: (r) => r.metrics?.piotroski_f_score,
    id: "f_score",
    header: "F-Score",
    cell: ({ row }) => row.original.metrics ? formatNumber(row.original.metrics.piotroski_f_score, 0) : "—",
  },
  {
    accessorFn: (r) => r.news_item.source,
    id: "source",
    header: "Source",
    cell: ({ row }) => {
      const url = row.original.news_item.url;
      return url ? (
        <a href={url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs truncate max-w-[100px] block">
          {row.original.news_item.source}
        </a>
      ) : (
        <span className="text-xs">{row.original.news_item.source}</span>
      );
    },
  },
  {
    accessorFn: (r) => r.news_item.title,
    id: "title",
    header: "Title",
    cell: ({ row }) => <span className="text-xs truncate max-w-[200px] block">{row.original.news_item.title}</span>,
  },
];

interface SignalsTableProps {
  signals: TradingSignal[];
  onDownload?: () => void;
}

export function SignalsTable({ signals, onDownload }: SignalsTableProps) {
  return (
    <DataTable
      data={signals}
      columns={columns}
      searchPlaceholder="Search tickers, sources…"
      onDownload={onDownload}
      compact
    />
  );
}
