"use client";

import { useState } from "react";
import { ScreenerConfigForm } from "@/components/forms/screener-config";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { NIFTY_50_TICKERS } from "@/lib/constants";
import { useScreener } from "@/hooks/use-screener";
import { formatCurrency, formatPct } from "@/lib/utils";
import type { ScreenerConfig, RiskConfig } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Search } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";

export default function INDScreenerPage() {
  const [screenerCfg, setScreenerCfg] = useState<ScreenerConfig>({
    min_price: 10, min_avg_volume: 500000, min_beta: 0.5, workers: 4, volume_multiplier: 1.5, lookback_days: 30, index_mode: false,
  });
  const [riskCfg, setRiskCfg] = useState<RiskConfig>({
    total_capital: 500000, max_open_trades: 5, risk_per_trade_pct: 2, pullback_tolerance_pct: 3, min_rr_ratio: 2, stop_loss_method: "tighter",
  });

  const { screenAsync, isScreening, screenResult, screenError, execute, isExecuting } = useScreener();
  const [fireOrders, setFireOrders] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);

  const handleRun = async () => {
    setPipelineRunning(true);
    try {
      const result = await screenAsync({ screener: screenerCfg, risk: riskCfg, tickers: [] });
      if (fireOrders && result?.trade_plans?.length > 0) {
        execute(result.trade_plans);
      }
    } finally {
      setPipelineRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4">
          <h3 className="text-sm font-semibold">Screener Config</h3>
          <ScreenerConfigForm onConfigChange={setScreenerCfg} />

          <h3 className="text-sm font-semibold mt-4">Risk Config</h3>
          <div className="space-y-2">
            <div><Label className="text-xs">Total Capital</Label><Input type="number" className="h-8 text-sm" value={riskCfg.total_capital} onChange={(e) => setRiskCfg({ ...riskCfg, total_capital: Number(e.target.value) })} /></div>
            <div><Label className="text-xs">Max Open Trades</Label><Input type="number" className="h-8 text-sm" value={riskCfg.max_open_trades} onChange={(e) => setRiskCfg({ ...riskCfg, max_open_trades: Number(e.target.value) })} /></div>
            <div><Label className="text-xs">Risk per Trade (%)</Label><Input type="number" className="h-8 text-sm" value={riskCfg.risk_per_trade_pct} onChange={(e) => setRiskCfg({ ...riskCfg, risk_per_trade_pct: Number(e.target.value) })} /></div>
            <div><Label className="text-xs">Min R:R Ratio</Label><Input type="number" className="h-8 text-sm" value={riskCfg.min_rr_ratio} onChange={(e) => setRiskCfg({ ...riskCfg, min_rr_ratio: Number(e.target.value) })} /></div>
          </div>

          <Button className="w-full" onClick={handleRun} disabled={isScreening || pipelineRunning || isExecuting}>
            {(isScreening || pipelineRunning) ? "Screening…" : <><Search className="mr-1 h-4 w-4" /> Screen &amp; Verdict</>}
          </Button>
          <div className="flex items-center gap-2 mt-1">
            <Checkbox id="fire-orders" checked={fireOrders} onCheckedChange={(v) => setFireOrders(v === true)} />
            <Label htmlFor="fire-orders" className="text-xs cursor-pointer select-none">Fire orders after screening &amp; verdict</Label>
          </div>
          {screenError && <p className="text-sm text-destructive">{screenError}</p>}
        </div>

        <div className="md:col-span-3 space-y-4">
          {isScreening && <Spinner />}
          {screenResult && (
            <>
              <MetricsGrid>
                <MetricCard label="Stocks Found" value={screenResult.stocks.length} />
                <MetricCard label="Passed Filter" value={screenResult.stocks.filter((s) => s.passed).length} color="text-green-500" />
                <MetricCard label="Trade Plans" value={screenResult.trade_plans.length} />
              </MetricsGrid>

              {screenResult.stocks.length > 0 && (
                <div className="content-panel p-4">
                  <h4 className="text-sm font-semibold mb-2">Screened Stocks</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead><tr className="border-b text-muted-foreground text-left">
                        <th className="py-2 pr-3">Ticker</th><th className="py-2 pr-3">Price</th><th className="py-2 pr-3">Avg Volume</th><th className="py-2 pr-3">Beta</th><th className="py-2 pr-3">Score</th><th className="py-2">Passed</th>
                      </tr></thead>
                      <tbody>{screenResult.stocks.map((s) => (
                        <tr key={s.ticker} className="border-b">
                          <td className="py-2 pr-3 font-mono">{s.ticker}</td>
                          <td className="py-2 pr-3">{formatCurrency(s.price)}</td>
                          <td className="py-2 pr-3">{s.avg_volume.toLocaleString()}</td>
                          <td className="py-2 pr-3">{s.beta.toFixed(2)}</td>
                          <td className="py-2 pr-3">{s.score.toFixed(2)}</td>
                          <td className="py-2">{s.passed ? "✅" : "❌"}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                </div>
              )}

              {screenResult.trade_plans.length > 0 && (
                <div className="content-panel p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold">Trade Plans</h4>
                    <Button size="sm" onClick={() => execute(screenResult.trade_plans)} disabled={isExecuting}>
                      {isExecuting ? "Executing…" : "Execute All"}
                    </Button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead><tr className="border-b text-muted-foreground text-left">
                        <th className="py-2 pr-3">Ticker</th><th className="py-2 pr-3">Entry</th><th className="py-2 pr-3">SL</th><th className="py-2 pr-3">Target</th><th className="py-2 pr-3">Qty</th><th className="py-2">R:R</th>
                      </tr></thead>
                      <tbody>{screenResult.trade_plans.map((p) => (
                        <tr key={p.ticker} className="border-b">
                          <td className="py-2 pr-3 font-mono">{p.ticker}</td>
                          <td className="py-2 pr-3">{formatCurrency(p.entry_price)}</td>
                          <td className="py-2 pr-3 text-red-500">{formatCurrency(p.stop_loss)}</td>
                          <td className="py-2 pr-3 text-green-500">{formatCurrency(p.target_price)}</td>
                          <td className="py-2 pr-3">{p.quantity}</td>
                          <td className="py-2">{p.rr_ratio.toFixed(2)}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
