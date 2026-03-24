"use client";

import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "./data-table";
import { formatNumber, getZScoreStatus, getMScoreStatus, getFScoreStatus } from "@/lib/utils";
import { HEALTH_COLORS } from "@/lib/constants";

export interface FundamentalRow {
  ticker: string;
  price: number;
  z_score: number;
  z_status: string;
  m_score: number;
  m_status: string;
  f_score: number;
  f_status: string;
}

function StatusBadge({ status }: { status: string }) {
  const color = HEALTH_COLORS[status] || "#999";
  return (
    <span className="px-1.5 py-0.5 rounded text-xs font-medium" style={{ color }}>
      {status}
    </span>
  );
}

const columns: ColumnDef<FundamentalRow, unknown>[] = [
  { accessorKey: "ticker", header: "Stock" },
  { accessorKey: "price", header: "Price", cell: ({ row }) => `$${formatNumber(row.original.price)}` },
  { accessorKey: "z_score", header: "Z-Score", cell: ({ row }) => formatNumber(row.original.z_score, 2) },
  { accessorKey: "z_status", header: "Z Status", cell: ({ row }) => <StatusBadge status={row.original.z_status} /> },
  { accessorKey: "m_score", header: "M-Score", cell: ({ row }) => formatNumber(row.original.m_score, 2) },
  { accessorKey: "m_status", header: "M Status", cell: ({ row }) => <StatusBadge status={row.original.m_status} /> },
  { accessorKey: "f_score", header: "F-Score", cell: ({ row }) => formatNumber(row.original.f_score, 0) },
  { accessorKey: "f_status", header: "F Status", cell: ({ row }) => <StatusBadge status={row.original.f_status} /> },
];

interface FundamentalTableProps {
  data: FundamentalRow[];
}

export function FundamentalTable({ data }: FundamentalTableProps) {
  return <DataTable data={data} columns={columns} searchPlaceholder="Search stocks…" compact />;
}

/** Helper to build FundamentalRow[] from signals */
export function buildFundamentalRows(signals: { news_item: { ticker: string }; metrics: { current_price: number; altman_z_score: number; beneish_m_score: number; piotroski_f_score: number } | null }[]): FundamentalRow[] {
  const seen = new Set<string>();
  const rows: FundamentalRow[] = [];
  for (const s of signals) {
    if (!s.metrics || seen.has(s.news_item.ticker)) continue;
    seen.add(s.news_item.ticker);
    rows.push({
      ticker: s.news_item.ticker,
      price: s.metrics.current_price,
      z_score: s.metrics.altman_z_score,
      z_status: getZScoreStatus(s.metrics.altman_z_score),
      m_score: s.metrics.beneish_m_score,
      m_status: getMScoreStatus(s.metrics.beneish_m_score),
      f_score: s.metrics.piotroski_f_score,
      f_status: getFScoreStatus(s.metrics.piotroski_f_score),
    });
  }
  return rows;
}
