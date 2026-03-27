"use client";

import { useState, useCallback } from "react";
import { usStocksApi, type TradingSignal } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, getDecisionBadgeClass, getDecisionColor } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { Loader2, Play, BarChart3, TrendingUp, TrendingDown } from "lucide-react";
import { toast } from "sonner";

const DECISION_COLORS: Record<string, string> = {
  STRONG_BUY: "#00cc44", BUY: "#66ff99", HOLD: "#ffcc00", SELL: "#ff9933", STRONG_SELL: "#ff3333",
};

const DEFAULT_TICKERS = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, LT, KOTAKBANK, HINDUNILVR";

export default function IndAnalysisPage() {
  const [tickerText, setTickerText] = useState(DEFAULT_TICKERS);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [hasResults, setHasResults] = useState(false);

  const parseTickers = useCallback((): string[] => {
    return tickerText.split(/[,\n]+/).map((t) => {
      let ticker = t.trim().toUpperCase();
      if (ticker && !ticker.endsWith(".NS")) ticker += ".NS";
      return ticker;
    }).filter(Boolean);
  }, [tickerText]);

  const handleRun = async () => {
    const tickers = parseTickers();
    if (tickers.length === 0) { toast.error("Enter at least one ticker"); return; }
    setLoading(true);
    setProgress(10);
    try {
      setProgress(30);
      // Uses same analysis endpoint with IND tickers (.NS suffix)
      const res = await usStocksApi.analyze(tickers);
      setProgress(100);
      if (res.success && res.signals) {
        setSignals(res.signals);
        setHasResults(true);
        toast.success(`Analysis complete — ${res.signal_count} signals`);
      }
    } catch (err: any) {
      toast.error(err?.message || "Analysis failed");
    } finally { setLoading(false); setProgress(0); }
  };

  const decisionCounts = signals.reduce((acc, s) => {
    const d = s.decision?.toUpperCase().replace(/\s+/g, "_") || "HOLD";
    acc[d] = (acc[d] || 0) + 1; return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(decisionCounts).map(([name, value]) => ({
    name: name.replace(/_/g, " "), value, color: DECISION_COLORS[name] || "#666",
  }));

  const buyCount = signals.filter((s) => s.decision?.toUpperCase().includes("BUY")).length;
  const sellCount = signals.filter((s) => s.decision?.toUpperCase().includes("SELL")).length;

  const columns: ColumnDef<TradingSignal>[] = [
    { accessorKey: "ticker", header: ({ column }) => <SortableHeader column={column} title="Ticker" />, cell: ({ row }) => <span className="font-mono font-bold">{(row.getValue("ticker") as string).replace(".NS","")}</span> },
    { accessorKey: "decision", header: ({ column }) => <SortableHeader column={column} title="Decision" />, cell: ({ row }) => <Badge className={getDecisionBadgeClass(row.getValue("decision") as string)}>{row.getValue("decision")}</Badge> },
    { accessorKey: "decision_score", header: ({ column }) => <SortableHeader column={column} title="Score" />, cell: ({ row }) => <span className={getDecisionColor(row.original.decision)}>{formatNumber(row.getValue("decision_score") as number, 3)}</span> },
    { accessorKey: "current_price", header: "Price", cell: ({ row }) => `₹${formatNumber(row.getValue("current_price") as number)}` },
    { accessorKey: "rsi", header: "RSI", cell: ({ row }) => formatNumber(row.getValue("rsi") as number, 1) },
    { accessorKey: "sentiment_label", header: "Sentiment", cell: ({ row }) => row.getValue("sentiment_label") ? <Badge variant="outline" className="capitalize">{row.getValue("sentiment_label")}</Badge> : "—" },
    { accessorKey: "source", header: "Source" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">NSE Stock Analysis</h1>
        <p className="text-sm text-muted-foreground">Indian market analysis — news, sentiment, fundamental & technical signals</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">NSE Ticker Selection</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Enter NSE tickers (comma-separated, .NS suffix added automatically)</Label>
              <Textarea value={tickerText} onChange={(e) => setTickerText(e.target.value)} rows={3} />
              <p className="text-xs text-muted-foreground">{parseTickers().length} ticker(s)</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Run Analysis</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handleRun} disabled={loading || parseTickers().length === 0} className="w-full" size="lg">
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {loading ? "Analyzing..." : "Run Analysis"}
            </Button>
            {loading && <Progress value={progress} />}
          </CardContent>
        </Card>
      </div>

      {hasResults && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCard title="Total Signals" value={String(signals.length)} icon={<BarChart3 className="h-4 w-4" />} />
            <MetricCard title="Buy Signals" value={String(buyCount)} changeType="positive" icon={<TrendingUp className="h-4 w-4" />} />
            <MetricCard title="Sell Signals" value={String(sellCount)} changeType="negative" icon={<TrendingDown className="h-4 w-4" />} />
            <MetricCard title="Hold" value={String(signals.length - buyCount - sellCount)} />
          </div>

          <Tabs defaultValue="overview">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="table">Detailed Table</TabsTrigger>
              <TabsTrigger value="top">Top Signals</TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <Card>
                <CardContent className="pt-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} innerRadius={50} dataKey="value" label={({ name, value }) => `${name} (${value})`}>
                        {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Pie>
                      <Tooltip /><Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="table">
              <Card><CardContent className="pt-6"><DataTable columns={columns} data={signals} searchKey="ticker" /></CardContent></Card>
            </TabsContent>
            <TabsContent value="top">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle className="text-base text-profit">Top Buy</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    {signals.filter((s) => s.decision?.toUpperCase().includes("BUY")).sort((a, b) => b.decision_score - a.decision_score).slice(0, 5).map((s) => (
                      <div key={`${s.ticker}-${s.source}`} className="flex items-center justify-between rounded border p-2">
                        <span className="font-mono text-sm font-bold">{s.ticker.replace(".NS","")}</span>
                        <Badge className="badge-buy">{formatNumber(s.decision_score, 3)}</Badge>
                      </div>
                    ))}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle className="text-base text-loss">Top Sell</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    {signals.filter((s) => s.decision?.toUpperCase().includes("SELL")).sort((a, b) => a.decision_score - b.decision_score).slice(0, 5).map((s) => (
                      <div key={`${s.ticker}-${s.source}`} className="flex items-center justify-between rounded border p-2">
                        <span className="font-mono text-sm font-bold">{s.ticker.replace(".NS","")}</span>
                        <Badge className="badge-sell">{formatNumber(s.decision_score, 3)}</Badge>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
