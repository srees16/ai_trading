"use client";

import { useState, useCallback, lazy, Suspense } from "react";
import { indStocksApi, type TradingSignal } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, getDecisionBadgeClass, getDecisionColor, downloadCsv } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { Loader2, Play, Upload, BarChart3, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { toast } from "sonner";

// Lazy-load recharts — only downloaded when charts are rendered (saves ~370 KB)
const DecisionPieChart = lazy(() => import("recharts").then(mod => ({
  default: function DecisionPieChartInner({ data }: { data: { name: string; value: number; color: string }[] }) {
    const { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } = mod;
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" outerRadius={100} innerRadius={50} dataKey="value" label={({ name, value }: { name: string; value: number }) => `${name} (${value})`}>
            {data.map((e, i) => <Cell key={i} fill={e.color} />)}
          </Pie>
          <Tooltip /><Legend />
        </PieChart>
      </ResponsiveContainer>
    );
  }
})));

const ScoreScatterChart = lazy(() => import("recharts").then(mod => ({
  default: function ScoreScatterChartInner({ data }: { data: { ticker: string; score: number; sentiment: number; decision: string }[] }) {
    const { ScatterChart, Scatter, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } = mod;
    return (
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
          <XAxis dataKey="ticker" type="category" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
          <YAxis dataKey="score" stroke="hsl(215, 20%, 55%)" />
          <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
          <Scatter data={data} fill="#00cc44" />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }
})));

const SentimentBarChart = lazy(() => import("recharts").then(mod => ({
  default: function SentimentBarChartInner({ data }: { data: { ticker: string; confidence: number; label: string; fill: string }[] }) {
    const { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } = mod;
    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
          <XAxis dataKey="ticker" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
          <YAxis stroke="hsl(215, 20%, 55%)" />
          <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
          <Bar dataKey="confidence" name="Confidence %" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }
})));

const DECISION_COLORS: Record<string, string> = {
  STRONG_BUY: "#00cc44", BUY: "#66ff99", HOLD: "#ffcc00", SELL: "#ff9933", STRONG_SELL: "#ff3333",
};

const DEFAULT_TICKERS = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, LT, KOTAKBANK, HINDUNILVR";

export default function IndAnalysisPage() {
  const [inputMode, setInputMode] = useState<"manual" | "csv">("manual");
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

  const handleCsvUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      const tickers = text
        .split(/[,\n]+/)
        .map((t) => t.trim().replace(/"/g, ""))
        .filter((t) => t && t !== "ticker" && t !== "symbol");
      setTickerText(tickers.join(", "));
      setInputMode("manual");
      toast.success(`Loaded ${tickers.length} tickers from CSV`);
    };
    reader.readAsText(file);
  };

  const handleRun = async () => {
    const tickers = parseTickers();
    if (tickers.length === 0) { toast.error("Enter at least one ticker"); return; }
    setLoading(true);
    setProgress(10);
    try {
      setProgress(30);
      const res = await indStocksApi.analyze(tickers);
      setProgress(100);
      if (res.success && res.signals) {
        setSignals(res.signals);
        setHasResults(true);
        toast.success(`Analysis complete — ${res.signal_count} signals`);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      toast.error(message);
    } finally { setLoading(false); setProgress(0); }
  };

  const decisionCounts = signals.reduce((acc, s) => {
    const d = s.decision?.toUpperCase().replace(/\s+/g, "_") || "HOLD";
    acc[d] = (acc[d] || 0) + 1; return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(decisionCounts).map(([name, value]) => ({
    name: name.replace(/_/g, " "), value, color: DECISION_COLORS[name] || "#666",
  }));

  const scatterData = signals.map((s) => ({
    ticker: s.ticker, score: s.decision_score, sentiment: s.sentiment_score || 0, decision: s.decision,
  }));

  const sentimentData = signals
    .filter((s) => s.sentiment_confidence != null)
    .map((s) => ({
      ticker: s.ticker,
      confidence: (s.sentiment_confidence || 0) * 100,
      label: s.sentiment_label || "neutral",
      fill: s.sentiment_label === "positive" ? "#00cc44" : s.sentiment_label === "negative" ? "#ff3333" : "#ffcc00",
    }));

  const buyCount = signals.filter((s) => s.decision?.toUpperCase().includes("BUY")).length;
  const sellCount = signals.filter((s) => s.decision?.toUpperCase().includes("SELL")).length;
  const avgScore = signals.length > 0 ? signals.reduce((a, s) => a + s.decision_score, 0) / signals.length : 0;

  const handleExport = () => {
    downloadCsv(
      ["Ticker", "Decision", "Score", "Price", "RSI", "Sentiment", "Source"],
      signals.map((s) => [s.ticker, s.decision, s.decision_score, s.current_price, s.rsi, s.sentiment_label, s.source]),
      "ind_stock_analysis.csv"
    );
  };

  const columns: ColumnDef<TradingSignal>[] = [
    { accessorKey: "ticker", header: ({ column }) => <SortableHeader column={column} title="Ticker" />, cell: ({ row }) => <span className="font-mono font-bold">{(row.getValue("ticker") as string).replace(".NS","")}</span> },
    { accessorKey: "decision", header: ({ column }) => <SortableHeader column={column} title="Decision" />, cell: ({ row }) => <Badge className={getDecisionBadgeClass(row.getValue("decision") as string)}>{row.getValue("decision")}</Badge> },
    { accessorKey: "decision_score", header: ({ column }) => <SortableHeader column={column} title="Score" />, cell: ({ row }) => <span className={getDecisionColor(row.original.decision)}>{formatNumber(row.getValue("decision_score") as number, 3)}</span> },
    { accessorKey: "current_price", header: "Price", cell: ({ row }) => `₹${formatNumber(row.getValue("current_price") as number)}` },
    { accessorKey: "rsi", header: "RSI", cell: ({ row }) => formatNumber(row.getValue("rsi") as number, 1) },
    { accessorKey: "sentiment_label", header: "Sentiment", cell: ({ row }) => row.getValue("sentiment_label") ? <Badge variant="outline" className="capitalize">{row.getValue("sentiment_label")}</Badge> : "—" },
    { accessorKey: "source", header: "Source", cell: ({ row }) => <span className="text-muted-foreground">{row.getValue("source")}</span> },
    { accessorKey: "reasoning", header: "Reasoning", cell: ({ row }) => <span className="max-w-xs truncate text-muted-foreground" title={row.getValue("reasoning") as string}>{(row.getValue("reasoning") as string)?.slice(0, 60)}…</span> },
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
            <div className="flex gap-2">
              <Button variant={inputMode === "manual" ? "default" : "outline"} size="sm" onClick={() => setInputMode("manual")}>Manual Entry</Button>
              <Button variant={inputMode === "csv" ? "default" : "outline"} size="sm" onClick={() => setInputMode("csv")}><Upload className="mr-1.5 h-3.5 w-3.5" /> Upload CSV</Button>
            </div>
            {inputMode === "manual" ? (
              <div className="space-y-2">
                <Label>Enter NSE tickers (comma-separated, .NS suffix added automatically)</Label>
                <Textarea value={tickerText} onChange={(e) => setTickerText(e.target.value)} rows={3} />
                <p className="text-xs text-muted-foreground">{parseTickers().length} ticker(s)</p>
              </div>
            ) : (
              <div className="space-y-2">
                <Label>Upload CSV file with ticker symbols</Label>
                <Input type="file" accept=".csv" onChange={handleCsvUpload} />
              </div>
            )}
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
            <MetricCard title="Avg Score" value={formatNumber(avgScore, 3)} changeType={avgScore > 0 ? "positive" : avgScore < 0 ? "negative" : "neutral"} icon={<Minus className="h-4 w-4" />} />
          </div>

          <Tabs defaultValue="overview">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="table">Detailed Table</TabsTrigger>
              <TabsTrigger value="top">Top Signals</TabsTrigger>
              <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle className="text-base">Decision Distribution</CardTitle></CardHeader>
                  <CardContent>
                    <Suspense fallback={<div className="h-[300px] animate-pulse rounded bg-muted" />}>
                      <DecisionPieChart data={pieData} />
                    </Suspense>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle className="text-base">Score Distribution</CardTitle></CardHeader>
                  <CardContent>
                    <Suspense fallback={<div className="h-[300px] animate-pulse rounded bg-muted" />}>
                      <ScoreScatterChart data={scatterData} />
                    </Suspense>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            <TabsContent value="table">
              <Card><CardContent className="pt-6">
                <DataTable columns={columns} data={signals} searchKey="ticker" onExport={handleExport} />
              </CardContent></Card>
            </TabsContent>
            <TabsContent value="top">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle className="text-base text-profit">Top Buy</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    {signals.filter((s) => s.decision?.toUpperCase().includes("BUY")).sort((a, b) => b.decision_score - a.decision_score).slice(0, 5).map((s) => (
                      <div key={`${s.ticker}-${s.source}`} className="flex items-center justify-between rounded border p-2">
                        <div>
                          <span className="font-mono text-sm font-bold">{s.ticker.replace(".NS","")}</span>
                          <p className="text-xs text-muted-foreground">{s.reasoning?.slice(0, 80)}</p>
                        </div>
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
                        <div>
                          <span className="font-mono text-sm font-bold">{s.ticker.replace(".NS","")}</span>
                          <p className="text-xs text-muted-foreground">{s.reasoning?.slice(0, 80)}</p>
                        </div>
                        <Badge className="badge-sell">{formatNumber(s.decision_score, 3)}</Badge>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            <TabsContent value="sentiment">
              <Card>
                <CardHeader><CardTitle className="text-base">Sentiment Confidence by Ticker</CardTitle></CardHeader>
                <CardContent>
                  <Suspense fallback={<div className="h-[400px] animate-pulse rounded bg-muted" />}>
                    <SentimentBarChart data={sentimentData} />
                  </Suspense>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
