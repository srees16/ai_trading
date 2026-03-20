"use client";

import { useState, useEffect } from "react";
import { pipelineApi, type PipelineRunRecord, type LatestRunResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/shared/metric-card";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { formatNumber, downloadCsv } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { RefreshCw, Loader2, Clock, ScanSearch, TrendingUp, TrendingDown } from "lucide-react";
import { toast } from "sonner";

export default function IndHistoryPage() {
  const [loading, setLoading] = useState(false);
  const [latestRun, setLatestRun] = useState<LatestRunResponse | null>(null);
  const [runs, setRuns] = useState<PipelineRunRecord[]>([]);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [latest, history] = await Promise.all([
        pipelineApi.latest().catch(() => null),
        pipelineApi.history(100).catch(() => ({ success: false, runs: [], total: 0 })),
      ]);
      setLatestRun(latest);
      if (history.success) setRuns(history.runs);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to fetch history";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const runColumns: ColumnDef<PipelineRunRecord>[] = [
    { accessorKey: "id", header: ({ column }) => <SortableHeader column={column} title="ID" /> },
    { accessorKey: "run_type", header: "Type", cell: ({ row }) => <Badge variant="outline">{row.getValue("run_type")}</Badge> },
    { accessorKey: "timestamp", header: ({ column }) => <SortableHeader column={column} title="Timestamp" />, cell: ({ row }) => {
      const t = row.getValue("timestamp") as string;
      return t ? new Date(t).toLocaleString() : "—";
    }},
    { accessorKey: "universe_size", header: "Universe" },
    { accessorKey: "screened_count", header: "Screened" },
    { accessorKey: "buy_signals", header: "Buy", cell: ({ row }) => <span className="text-profit font-semibold">{row.getValue("buy_signals")}</span> },
    { accessorKey: "sell_signals", header: "Sell", cell: ({ row }) => <span className="text-loss font-semibold">{row.getValue("sell_signals")}</span> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => {
      const s = row.getValue("status") as string;
      return <Badge className={s === "success" ? "bg-profit/20 text-profit" : "bg-loss/20 text-loss"}>{s}</Badge>;
    }},
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Screener History</h1>
          <p className="text-sm text-muted-foreground">Pipeline run history and latest screening results</p>
        </div>
        <Button onClick={fetchAll} disabled={loading} variant="outline" size="sm">
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
          Refresh
        </Button>
      </div>

      {latestRun && latestRun.status !== "no_data" && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
            <MetricCard title="Run Type" value={latestRun.run_type || "—"} icon={<ScanSearch className="h-4 w-4" />} />
            <MetricCard title="Status" value={latestRun.status || "—"} />
            <MetricCard title="Universe" value={String(latestRun.universe_size)} />
            <MetricCard title="Screened" value={String(latestRun.screened_count)} />
            <MetricCard title="Buy Signals" value={String(latestRun.buy_signals)} changeType="positive" icon={<TrendingUp className="h-4 w-4" />} />
            <MetricCard title="Sell Signals" value={String(latestRun.sell_signals)} changeType="negative" icon={<TrendingDown className="h-4 w-4" />} />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Clock className="h-4 w-4" />
                Latest Run
                <Badge variant="outline">{latestRun.status}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="text-muted-foreground">Timestamp:</span> <span className="font-mono">{latestRun.timestamp || "—"}</span></div>
                <div><span className="text-muted-foreground">Run Type:</span> <span className="font-bold">{latestRun.run_type || "—"}</span></div>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      <Card>
        <CardHeader><CardTitle className="text-base">All Pipeline Runs ({runs.length})</CardTitle></CardHeader>
        <CardContent>
          {runs.length > 0 ? (
            <DataTable
              columns={runColumns}
              data={runs}
              searchKey="run_type"
              onExport={() => downloadCsv(
                ["ID", "Type", "Timestamp", "Universe", "Screened", "Buy", "Sell", "Status"],
                runs.map((r) => [r.id, r.run_type, r.timestamp, r.universe_size, r.screened_count, r.buy_signals, r.sell_signals, r.status]),
                "pipeline_history.csv"
              )}
            />
          ) : !loading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <ScanSearch className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No screening runs found</p>
              <p className="text-xs text-muted-foreground mt-1">Run a screening pipeline to see results here</p>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
