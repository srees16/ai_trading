"use client";

import { useState, useEffect } from "react";
import { usStocksApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { LoadingSpinner } from "@/components/shared/metric-card";
import { Badge } from "@/components/ui/badge";
import { formatNumber, getDecisionBadgeClass } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { History, Calendar } from "lucide-react";

const PERIODS = [
  { label: "Last 7 days", value: "7" },
  { label: "Last 14 days", value: "14" },
  { label: "Last 30 days", value: "30" },
  { label: "Last 90 days", value: "90" },
];

export default function USHistoryPage() {
  const [period, setPeriod] = useState("30");
  const [runs, setRuns] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    usStocksApi
      .history(100)
      .then((res) => setRuns(res.runs || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [period]);

  const columns: ColumnDef<Record<string, unknown>>[] = [
    {
      accessorKey: "id",
      header: ({ column }) => <SortableHeader column={column} title="Run ID" />,
    },
    {
      accessorKey: "market",
      header: "Market",
      cell: ({ row }) => <Badge variant="outline">{(row.getValue("market") as string) || "US"}</Badge>,
    },
    {
      accessorKey: "ticker_count",
      header: "Tickers",
    },
    {
      accessorKey: "signal_count",
      header: "Signals",
    },
    {
      accessorKey: "created_at",
      header: ({ column }) => <SortableHeader column={column} title="Date" />,
      cell: ({ row }) => {
        const d = row.getValue("created_at") as string;
        return d ? new Date(d).toLocaleString() : "—";
      },
    },
  ];

  if (loading) return <LoadingSpinner size="lg" className="h-[60vh]" />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Analysis History</h1>
          <p className="text-sm text-muted-foreground">Previous analysis runs and trading signals</p>
        </div>
        <Select value={period} onValueChange={setPeriod}>
          <SelectTrigger className="w-40">
            <Calendar className="mr-2 h-4 w-4" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PERIODS.map((p) => (
              <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Tabs defaultValue="runs">
        <TabsList>
          <TabsTrigger value="runs">Analysis Runs</TabsTrigger>
        </TabsList>
        <TabsContent value="runs">
          <Card>
            <CardContent className="pt-6">
              <DataTable
                columns={columns}
                data={runs}
                searchKey="id"
                searchPlaceholder="Search runs..."
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
