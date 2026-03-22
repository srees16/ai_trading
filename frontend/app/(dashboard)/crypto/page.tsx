"use client";

import { useState } from "react";
import { TickerInput } from "@/components/forms/ticker-input";
import { SignalsTable } from "@/components/tables/signals-table";
import { TopSignals } from "@/components/tables/top-signals";
import { DecisionPieChart } from "@/components/charts/decision-pie-chart";
import { SentimentBarChart } from "@/components/charts/sentiment-bar-chart";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { useAnalysis } from "@/hooks/use-analysis";
import { DEFAULT_CRYPTO_TICKERS } from "@/lib/constants";
import { decisionSummary } from "@/lib/utils";
import { Play, Bitcoin } from "lucide-react";

export default function CryptoPage() {
  const [tickers, setTickers] = useState<string[]>(DEFAULT_CRYPTO_TICKERS);
  const { run, isRunning, result, error } = useAnalysis("US"); // crypto uses US market API

  const handleRun = () => {
    if (tickers.length > 0) run({ tickers, market: "US" });
  };

  const signals = result?.signals ?? [];
  const summary = result?.summary ?? decisionSummary([]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Bitcoin className="h-5 w-5 text-amber-500" />
        <h2 className="text-lg font-semibold">Crypto Analysis</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4 min-w-0 overflow-hidden">
          <TickerInput defaultTickers={DEFAULT_CRYPTO_TICKERS} onTickersChange={setTickers} />
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
                <MetricCard label="Total" value={summary.total} />
                <MetricCard label="Strong Buy" value={summary.strong_buy} color="text-green-500" />
                <MetricCard label="Buy" value={summary.buy} color="text-emerald-400" />
                <MetricCard label="Sell" value={summary.sell} color="text-orange-500" />
                <MetricCard label="Strong Sell" value={summary.strong_sell} color="text-red-500" />
              </MetricsGrid>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="content-panel p-4"><DecisionPieChart signals={signals} /></div>
                <div className="content-panel p-4"><SentimentBarChart signals={signals} /></div>
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
