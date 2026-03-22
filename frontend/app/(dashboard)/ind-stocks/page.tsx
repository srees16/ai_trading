"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { TickerInput } from "@/components/forms/ticker-input";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { useAnalysis } from "@/hooks/use-analysis";
import { DEFAULT_IND_TICKERS, NIFTY_50_TICKERS } from "@/lib/constants";
import { decisionSummary } from "@/lib/utils";
import { Play } from "lucide-react";

const SignalsTable = dynamic(() => import("@/components/tables/signals-table").then(m => ({ default: m.SignalsTable })), { ssr: false });
const TopSignals = dynamic(() => import("@/components/tables/top-signals").then(m => ({ default: m.TopSignals })), { ssr: false });
const DecisionPieChart = dynamic(() => import("@/components/charts/decision-pie-chart").then(m => ({ default: m.DecisionPieChart })), { ssr: false });
const ScoreScatterChart = dynamic(() => import("@/components/charts/score-scatter-chart").then(m => ({ default: m.ScoreScatterChart })), { ssr: false });
const SentimentBarChart = dynamic(() => import("@/components/charts/sentiment-bar-chart").then(m => ({ default: m.SentimentBarChart })), { ssr: false });

export default function INDStocksPage() {
  const [tickers, setTickers] = useState<string[]>(DEFAULT_IND_TICKERS);
  const { run, isRunning, result, error } = useAnalysis("IND");

  const handleRun = () => {
    if (tickers.length > 0) run({ tickers, market: "IND" });
  };

  const signals = result?.signals ?? [];
  const summary = result?.summary ?? decisionSummary([]);

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold">IND Stocks Analysis</h3>
          <TickerInput defaultTickers={DEFAULT_IND_TICKERS} onTickersChange={setTickers} />
          <Button className="w-full" onClick={handleRun} disabled={isRunning}>
            {isRunning ? "Analyzing…" : <><Play className="mr-1 h-4 w-4" /> Run Analysis</>}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <div className="md:col-span-3 space-y-4">
          {isRunning && <Spinner />}
          {signals.length > 0 && (
            <>
              <MetricsGrid>
                <MetricCard label="Total Signals" value={summary.total} />
                <MetricCard label="Strong Buy" value={summary.strong_buy} color="text-green-500" />
                <MetricCard label="Buy" value={summary.buy} color="text-emerald-400" />
                <MetricCard label="Hold" value={summary.hold} color="text-amber-400" />
                <MetricCard label="Sell" value={summary.sell} color="text-orange-500" />
                <MetricCard label="Strong Sell" value={summary.strong_sell} color="text-red-500" />
              </MetricsGrid>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="content-panel p-4"><DecisionPieChart signals={signals} /></div>
                <div className="content-panel p-4"><SentimentBarChart signals={signals} /></div>
                <div className="content-panel p-4"><ScoreScatterChart signals={signals} /></div>
              </div>

              <TopSignals signals={signals} />
              <div className="content-panel p-4"><SignalsTable signals={signals} /></div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
