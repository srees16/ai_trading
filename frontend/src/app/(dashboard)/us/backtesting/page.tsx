"use client";

import { useState, useEffect } from "react";
import { usStocksApi, type BacktestResponse, type StrategyInfo } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MetricCard } from "@/components/shared/metric-card";
import { LoadingSpinner } from "@/components/shared/metric-card";
import { TradingChart } from "@/components/shared/trading-chart";
import { formatNumber, formatPercent } from "@/lib/utils";
import { Loader2, Play, TrendingUp, BarChart3, Activity, Shield } from "lucide-react";
import { toast } from "sonner";

const PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"];

export default function USBacktestingPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL");
  const [period, setPeriod] = useState("1y");
  const [capital, setCapital] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResponse[]>([]);
  const [strategyLoading, setStrategyLoading] = useState(true);

  useEffect(() => {
    usStocksApi
      .strategies()
      .then((res) => {
        if (res.strategies) {
          setStrategies(res.strategies);
          if (res.strategies.length > 0) setSelectedStrategy(res.strategies[0].id);
        }
      })
      .catch(() => toast.error("Failed to load strategies"))
      .finally(() => setStrategyLoading(false));
  }, []);

  const handleRun = async () => {
    if (!selectedStrategy) return;
    const tickerList = tickers.split(/[,\n]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickerList.length === 0) {
      toast.error("Enter at least one ticker");
      return;
    }

    setLoading(true);
    try {
      const res = await usStocksApi.backtest({
        strategy_id: selectedStrategy,
        tickers: tickerList,
        parameters: { period },
        initial_capital: capital,
      });
      if (res.success) {
        setResults((prev) => [...prev, res]);
        toast.success("Backtest complete");
      } else {
        toast.error(res.error_message || "Backtest failed");
      }
    } catch (err: any) {
      toast.error(err?.message || "Backtest failed");
    } finally {
      setLoading(false);
    }
  };

  const currentStrategy = strategies.find((s) => s.id === selectedStrategy);
  const latestResult = results.length > 0 ? results[results.length - 1] : null;
  const metrics = latestResult?.metrics || {};

  if (strategyLoading) return <LoadingSpinner size="lg" className="h-[60vh]" />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Backtesting</h1>
        <p className="text-sm text-muted-foreground">Test trading strategies against historical data</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        {/* Config Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Strategy</Label>
              <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                <SelectTrigger>
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  {strategies.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {currentStrategy && (
                <p className="text-xs text-muted-foreground">{currentStrategy.description}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label>Ticker(s)</Label>
              <Input value={tickers} onChange={(e) => setTickers(e.target.value)} />
            </div>

            <div className="space-y-2">
              <Label>Period</Label>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PERIODS.map((p) => (
                    <SelectItem key={p} value={p}>{p}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Initial Capital ($)</Label>
              <Input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                min={1000}
                step={1000}
              />
            </div>

            <Button onClick={handleRun} disabled={loading} className="w-full">
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {loading ? "Running..." : "Run Backtest"}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="space-y-6 lg:col-span-3">
          {latestResult ? (
            <>
              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
                <MetricCard
                  title="Total Return"
                  value={formatPercent(metrics.total_return as number)}
                  changeType={(metrics.total_return as number) >= 0 ? "positive" : "negative"}
                  icon={<TrendingUp className="h-4 w-4" />}
                />
                <MetricCard title="Sharpe Ratio" value={formatNumber(metrics.sharpe_ratio as number)} icon={<BarChart3 className="h-4 w-4" />} />
                <MetricCard
                  title="Max Drawdown"
                  value={formatPercent(metrics.max_drawdown as number)}
                  changeType="negative"
                  icon={<Activity className="h-4 w-4" />}
                />
                <MetricCard title="Win Rate" value={formatPercent(metrics.win_rate as number)} icon={<Shield className="h-4 w-4" />} />
                <MetricCard title="Total Trades" value={String(metrics.total_trades || 0)} />
                <MetricCard
                  title="Final Value"
                  value={`$${formatNumber(metrics.final_value as number || capital)}`}
                  changeType={(metrics.total_return as number) >= 0 ? "positive" : "negative"}
                />
              </div>

              {/* Charts & Tables */}
              <Tabs defaultValue="charts">
                <TabsList>
                  <TabsTrigger value="charts">Charts</TabsTrigger>
                  <TabsTrigger value="data">Trade Data</TabsTrigger>
                </TabsList>
                <TabsContent value="charts">
                  <Card>
                    <CardContent className="pt-6">
                      {latestResult.charts?.map((chart: any, i: number) => (
                        <div key={i} className="mb-4">
                          {chart.type === "plotly" && chart.data && (
                            <div className="rounded border p-4 text-center text-sm text-muted-foreground">
                              Plotly chart: {chart.title || `Chart ${i + 1}`}
                            </div>
                          )}
                          {chart.type === "image" && chart.data && (
                            <img
                              src={`data:image/png;base64,${chart.data}`}
                              alt={chart.title || "Backtest chart"}
                              className="max-w-full rounded"
                            />
                          )}
                        </div>
                      ))}
                      {(!latestResult.charts || latestResult.charts.length === 0) && (
                        <p className="py-8 text-center text-muted-foreground">No charts available</p>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="data">
                  <Card>
                    <CardContent className="pt-6">
                      {latestResult.tables?.map((table: any, i: number) => (
                        <div key={i} className="mb-6">
                          <h4 className="mb-2 text-sm font-semibold">{table.title || `Table ${i + 1}`}</h4>
                          <pre className="max-h-96 overflow-auto rounded bg-muted p-4 text-xs">
                            {JSON.stringify(table.data || table, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </>
          ) : (
            <Card>
              <CardContent className="flex h-64 items-center justify-center">
                <div className="text-center">
                  <BarChart3 className="mx-auto h-12 w-12 text-muted-foreground/50" />
                  <p className="mt-2 text-muted-foreground">Select a strategy and run a backtest</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
