"use client";

import { useState } from "react";
import { BacktestConfig } from "@/components/forms/backtest-config";
import { EquityCurveChart } from "@/components/charts/equity-curve-chart";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { useBacktest } from "@/hooks/use-backtest";
import { DEFAULT_IND_TICKERS, NIFTY_50_TICKERS } from "@/lib/constants";
import { formatCurrency, formatPct } from "@/lib/utils";
import { Play } from "lucide-react";

export default function INDBacktestPage() {
  const { strategies, strategiesLoading, run, isRunning, result, error } = useBacktest("IND");
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const categories = [...new Set(strategies.map((s) => s.category))];

  const handleRun = () => {
    run({
      strategy_id: String(config.strategy_id ?? strategies[0]?.id ?? ""),
      tickers: String(config.tickers ?? DEFAULT_IND_TICKERS.join(",")).split(",").map((t) => t.trim()).filter(Boolean),
      params: (config.params as Record<string, number | boolean>) ?? {},
      initial_capital: Number(config.initial_capital ?? 100000),
      period: String(config.period ?? "1y"),
      start_date: config.start_date as string | undefined,
      end_date: config.end_date as string | undefined,
      market: "IND",
    });
  };

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4">
          <h3 className="text-sm font-semibold">IND Backtest</h3>
          {strategiesLoading ? <Spinner /> : (
            <BacktestConfig strategies={strategies} categories={categories} onConfigChange={(c) => setConfig(c as Record<string, unknown>)} defaultTickers={DEFAULT_IND_TICKERS.join(", ")} />
          )}
          <Button className="w-full" onClick={handleRun} disabled={isRunning || strategies.length === 0}>
            {isRunning ? "Running…" : <><Play className="mr-1 h-4 w-4" /> Run Backtest</>}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <div className="md:col-span-3 space-y-4">
          {isRunning && <Spinner />}
          {result && (
            <>
              <MetricsGrid>
                <MetricCard label="Total Return" value={formatPct(result.total_return)} color={result.total_return >= 0 ? "text-green-500" : "text-red-500"} />
                <MetricCard label="Sharpe Ratio" value={result.sharpe_ratio.toFixed(2)} />
                <MetricCard label="Max Drawdown" value={formatPct(result.max_drawdown)} color="text-red-500" />
                <MetricCard label="Win Rate" value={formatPct(result.win_rate)} />
                <MetricCard label="Total Trades" value={result.total_trades} />
                <MetricCard label="Final Value" value={formatCurrency(result.final_value, "INR")} />
              </MetricsGrid>
              {result.equity_curve.length > 0 && (
                <div className="content-panel p-4">
                  <h4 className="text-sm font-semibold mb-2">Equity Curve</h4>
                  <EquityCurveChart data={result.equity_curve} height={350} />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
