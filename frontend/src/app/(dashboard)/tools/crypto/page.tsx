"use client";

import { useState, useEffect } from "react";
import { cryptoApi, type CryptoStrategyInfo, type CryptoBacktestResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, formatPercent } from "@/lib/utils";
import { Loader2, Play, Bitcoin, TrendingUp, BarChart3, DollarSign } from "lucide-react";
import { toast } from "sonner";

const PERIODS = ["1y", "2y", "3y", "5y", "Custom"];

export default function CryptoPage() {
  const [strategies, setStrategies] = useState<CryptoStrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [tickers, setTickers] = useState("ETH, BTC, LTC");
  const [period, setPeriod] = useState("2y");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [capital, setCapital] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CryptoBacktestResponse | null>(null);
  const [cachedResults, setCachedResults] = useState<Record<string, CryptoBacktestResponse>>({});

  useEffect(() => {
    cryptoApi.strategies().then((res) => {
      if (res.success) {
        setStrategies(res.strategies);
        if (res.strategies.length > 0) setSelectedStrategy(res.strategies[0].id);
      }
    }).catch(() => {});
  }, []);

  const handleRun = async () => {
    const symbols = tickers.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (symbols.length < 2) { toast.error("Enter at least 2 crypto symbols"); return; }
    if (!selectedStrategy) { toast.error("Select a strategy"); return; }

    setLoading(true);
    try {
      const params: Record<string, unknown> = { period };
      const res = await cryptoApi.backtest({
        symbols,
        initial_capital: capital,
        parameters: params,
        ...(period === "Custom" && startDate ? { start_date: startDate } : {}),
        ...(period === "Custom" && endDate ? { end_date: endDate } : {}),
      });
      if (res.success) {
        setResult(res);
        setCachedResults((prev) => ({ ...prev, [selectedStrategy]: res }));
        toast.success("Backtest complete");
      }
    } catch (err: any) { toast.error(err?.message || "Backtest failed"); }
    finally { setLoading(false); }
  };

  const metrics = result?.metrics as Record<string, number> | undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Bitcoin className="h-6 w-6 text-amber-500" /> Crypto Strategies
        </h1>
        <p className="text-sm text-muted-foreground">Backtest crypto trading strategies across multiple digital assets</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Config Panel */}
        <Card>
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label className="text-xs">Strategy</Label>
              <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                <SelectTrigger><SelectValue placeholder="Select strategy" /></SelectTrigger>
                <SelectContent>{strategies.map((s) => <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>)}</SelectContent>
              </Select>
              {strategies.find((s) => s.id === selectedStrategy)?.description && (
                <p className="text-xs text-muted-foreground mt-1">{strategies.find((s) => s.id === selectedStrategy)?.description}</p>
              )}
            </div>
            <div>
              <Label className="text-xs">Crypto Ticker(s)</Label>
              <Input value={tickers} onChange={(e) => setTickers(e.target.value)} placeholder="ETH, BTC, LTC" />
              <p className="text-xs text-muted-foreground mt-1">Min 2 symbols, comma-separated</p>
            </div>
            <div>
              <Label className="text-xs">Data Period</Label>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>{PERIODS.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            {period === "Custom" && (
              <div className="grid grid-cols-2 gap-2">
                <div><Label className="text-xs">Start</Label><Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></div>
                <div><Label className="text-xs">End</Label><Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></div>
              </div>
            )}
            <div>
              <Label className="text-xs">Initial Capital ($)</Label>
              <Input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} min={100} step={1000} />
            </div>
            <Button onClick={handleRun} disabled={loading} className="w-full" size="lg">
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {loading ? "Running..." : "Run Backtest"}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {result && metrics ? (
            <>
              <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
                <MetricCard title="Total Return" value={formatPercent(metrics.total_return)} changeType={(metrics.total_return || 0) >= 0 ? "positive" : "negative"} icon={<TrendingUp className="h-4 w-4" />} />
                <MetricCard title="Sharpe Ratio" value={formatNumber(metrics.sharpe_ratio, 2)} />
                <MetricCard title="Sortino Ratio" value={formatNumber(metrics.sortino_ratio, 2)} />
                <MetricCard title="Max Drawdown" value={formatPercent(metrics.max_drawdown)} changeType="negative" />
                <MetricCard title="Total Trades" value={String(metrics.total_trades || 0)} icon={<BarChart3 className="h-4 w-4" />} />
                <MetricCard title="Final Value" value={`$${formatNumber(metrics.final_value)}`} icon={<DollarSign className="h-4 w-4" />} />
              </div>

              <Tabs defaultValue="charts">
                <TabsList>
                  <TabsTrigger value="charts">Charts</TabsTrigger>
                  <TabsTrigger value="data">Data</TabsTrigger>
                </TabsList>
                <TabsContent value="charts">
                  <Card><CardContent className="pt-6">
                    {result.charts?.length ? result.charts.map((chart, i) => (
                      <div key={i} className="mb-4">
                        {(chart as any).image && <img src={`data:image/png;base64,${(chart as any).image}`} alt={`Chart ${i + 1}`} className="w-full rounded" />}
                      </div>
                    )) : <p className="text-sm text-muted-foreground">No charts available</p>}
                  </CardContent></Card>
                </TabsContent>
                <TabsContent value="data">
                  <Card><CardContent className="pt-6">
                    <pre className="max-h-96 overflow-auto rounded bg-muted p-4 text-xs">{JSON.stringify(result.tables, null, 2)}</pre>
                  </CardContent></Card>
                </TabsContent>
              </Tabs>
            </>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-20">
                <Bitcoin className="h-16 w-16 text-muted-foreground/30 mb-4" />
                <p className="text-muted-foreground">Select a strategy and run a backtest</p>
                <p className="text-xs text-muted-foreground mt-1">Results will appear here</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
