"use client";

import { useState, useEffect } from "react";
import { pipelineApi, type LatestRunResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber } from "@/lib/utils";
import { RefreshCw, Loader2, Clock, ScanSearch, TrendingUp, TrendingDown } from "lucide-react";
import { toast } from "sonner";

export default function IndHistoryPage() {
  const [loading, setLoading] = useState(false);
  const [latestRun, setLatestRun] = useState<LatestRunResponse | null>(null);

  const fetchLatest = async () => {
    setLoading(true);
    try {
      const res = await pipelineApi.latest();
      setLatestRun(res);
    } catch (err: any) {
      toast.error(err?.message || "Failed to fetch history");
    } finally { setLoading(false); }
  };

  useEffect(() => { fetchLatest(); }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Screener History</h1>
          <p className="text-sm text-muted-foreground">Latest NSE screening and pipeline run results</p>
        </div>
        <Button onClick={fetchLatest} disabled={loading} variant="outline" size="sm">
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
          Refresh
        </Button>
      </div>

      {latestRun ? (
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
                Last Run Details
                <Badge variant="outline">{latestRun.status}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="text-muted-foreground">Timestamp:</span> <span className="font-mono">{latestRun.timestamp || "—"}</span></div>
                <div><span className="text-muted-foreground">Run Type:</span> <span className="font-bold">{latestRun.run_type || "—"}</span></div>
                <div><span className="text-muted-foreground">Universe Size:</span> {latestRun.universe_size}</div>
                <div><span className="text-muted-foreground">Screened:</span> {latestRun.screened_count}</div>
                <div><span className="text-muted-foreground">Buy Signals:</span> <span className="text-profit">{latestRun.buy_signals}</span></div>
                <div><span className="text-muted-foreground">Sell Signals:</span> <span className="text-loss">{latestRun.sell_signals}</span></div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : !loading ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <ScanSearch className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No screening runs found</p>
            <p className="text-xs text-muted-foreground mt-1">Run a screening pipeline to see results here</p>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
