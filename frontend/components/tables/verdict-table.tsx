"use client";

import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "./data-table";
import { DecisionBadge } from "@/components/common/decision-badge";
import { formatNumber } from "@/lib/utils";
import type { VerdictResult } from "@/lib/types";

const columns: ColumnDef<VerdictResult, unknown>[] = [
  { accessorKey: "ticker", header: "Ticker" },
  { accessorKey: "core_score", header: "Core", cell: ({ row }) => formatNumber(row.original.core_score, 1) },
  { accessorKey: "strategy_score", header: "Strategy", cell: ({ row }) => formatNumber(row.original.strategy_score, 1) },
  { accessorKey: "ml_score", header: "ML Features", cell: ({ row }) => formatNumber(row.original.ml_score, 1) },
  { accessorKey: "robustness_score", header: "Robustness", cell: ({ row }) => formatNumber(row.original.robustness_score, 1) },
  { accessorKey: "weighted_score", header: "Weighted", cell: ({ row }) => formatNumber(row.original.weighted_score, 1) },
  {
    accessorKey: "verdict",
    header: "Verdict",
    cell: ({ row }) => <DecisionBadge decision={row.original.verdict} />,
  },
];

interface VerdictTableProps {
  data: VerdictResult[];
}

export function VerdictTable({ data }: VerdictTableProps) {
  return <DataTable data={data} columns={columns} searchPlaceholder="Search tickers…" compact />;
}
