"use client";

import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "./data-table";
import { cn, formatNumber } from "@/lib/utils";
import type { LiveQuote } from "@/lib/types";

const columns: ColumnDef<LiveQuote, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol", size: 100 },
  {
    accessorKey: "ltp",
    header: "LTP",
    cell: ({ row }) => <span className="font-mono">{formatNumber(row.original.ltp)}</span>,
  },
  {
    accessorKey: "change",
    header: "Change",
    cell: ({ row }) => (
      <span className={row.original.change >= 0 ? "pnl-positive" : "pnl-negative"}>
        {row.original.change >= 0 ? "+" : ""}{formatNumber(row.original.change)}
      </span>
    ),
  },
  {
    accessorKey: "change_pct",
    header: "Change %",
    cell: ({ row }) => (
      <span className={row.original.change_pct >= 0 ? "pnl-positive" : "pnl-negative"}>
        {row.original.change_pct >= 0 ? "+" : ""}{row.original.change_pct.toFixed(2)}%
      </span>
    ),
  },
  { accessorKey: "open", header: "Open", cell: ({ row }) => formatNumber(row.original.open) },
  { accessorKey: "high", header: "High", cell: ({ row }) => formatNumber(row.original.high) },
  { accessorKey: "low", header: "Low", cell: ({ row }) => formatNumber(row.original.low) },
  { accessorKey: "close", header: "Close", cell: ({ row }) => formatNumber(row.original.close) },
  {
    accessorKey: "volume",
    header: "Volume",
    cell: ({ row }) => row.original.volume.toLocaleString(),
  },
];

interface LiveQuotesTableProps {
  quotes: LiveQuote[];
}

export function LiveQuotesTable({ quotes }: LiveQuotesTableProps) {
  return <DataTable data={quotes} columns={columns} searchPlaceholder="Search symbols…" compact pageSize={50} />;
}
