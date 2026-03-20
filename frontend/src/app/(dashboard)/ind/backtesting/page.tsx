"use client";

import { useState } from "react";
import { usStocksApi, type BacktestResponse, type StrategyInfo } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, formatPercent } from "@/lib/utils";
import { Loader2, Play, TrendingUp, BarChart3 } from "lucide-react";
import { toast } from "sonner";
import { useEffect } from "react";

export default function IndBacktestingPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [strategiesLoading, setStrategiesLoading] = useState(true);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [tickers, setTickers] = useState("RELIANCE.NS, TCS.NS, INFY.NS");
  const [period, setPeriod] = useState("2y");
  const [capital, setCapital] = useState(500000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResponse | null>(null);

  useEffect(() => {
    usStocksApi.strategies().then((res) => {
      if (res.success) {
        setStrategies(res.strategies);
        if (res.strategies.length > 0) setSelectedStrategy(res.strategies[0].id);
      }
    }).catch(() => {}).finally(() => setStrategiesLoading(false));
  }, []);

  const handleRun = async () => {
    const tickerList = tickers.split(",").map((t) => {
      let ticker = t.trim().toUpperCase();
      if (!ticker.endsWith(".NS")) ticker += ".NS";
      return ticker;
    }).filter(Boolean);
    if (tickerList.length === 0 || !selectedStrategy) { toast.error("Enter tickers and select a strategy"); return; }

    setLoading(true);
    try {
      const res = await usStocksApi.backtest({
        strategy_id: selectedStrategy,
        tickers: tickerList,
        parameters: { period },
        initial_capital: capital,
      });
      if (res.success) { setResult(res); toast.success("Backtest complete"); }
    } catch (err: any) { toast.error(err?.message || "Backtest failed"); }
    finally { setLoading(false); }
  };

  const metrics = result?.metrics as Record<string, number> | undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">NSE Backtesting</h1>
        <p className="text-sm text-muted-foreground">Backtest trading strategies on Indian market stocks</p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">Strategy & Tickers</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div>
              <Label className="text-xs">Strategy</Label>
              <Select value={selectedStrategy} onValueChange={setSelectedStrategy} disabled={strategiesLoading}>
                <SelectTrigger><SelectValue placeholder={strategiesLoading ? "Loading strategies…" : "Select strategy"} /></SelectTrigger>
                <SelectContent>{strategies.map((s) => <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">NSE Tickers (comma-separated, .NS added automatically)</Label>
              <Input value={tickers} onChange={(e) => setTickers(e.target.value)} />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Parameters</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div><Label className="text-xs">Period</Label>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="6mo">6 Months</SelectItem>
                  <SelectItem value="1y">1 Year</SelectItem>
                  <SelectItem value="2y">2 Years</SelectItem>
                  <SelectItem value="5y">5 Years</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div><Label className="text-xs">Capital (₹)</Label><Input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} step={50000} /></div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Run</CardTitle></CardHeader>
          <CardContent>
            <Button onClick={handleRun} disabled={loading} className="w-full" size="lg">
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {loading ? "Running..." : "Run Backtest"}
            </Button>
          </CardContent>
        </Card>
      </div>

      {result && metrics && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
            <MetricCard title="Total Return" value={formatPercent(metrics.total_return)} changeType={(metrics.total_return || 0) >= 0 ? "positive" : "negative"} icon={<TrendingUp className="h-4 w-4" />} />
            <MetricCard title="Sharpe Ratio" value={formatNumber(metrics.sharpe_ratio, 2)} />
            <MetricCard title="Sortino Ratio" value={formatNumber(metrics.sortino_ratio, 2)} />
            <MetricCard title="Max Drawdown" value={formatPercent(metrics.max_drawdown)} changeType="negative" />
            <MetricCard title="Total Trades" value={String(metrics.total_trades || 0)} icon={<BarChart3 className="h-4 w-4" />} />
            <MetricCard title="Final Value" value={`₹${formatNumber(metrics.final_value)}`} />
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
      )}
    </div>
  );
}
