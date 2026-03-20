"use client";

import { useState } from "react";
import { TickerInput } from "@/components/forms/ticker-input";
import { VerdictConfig } from "@/components/forms/verdict-config";
import { VerdictTable } from "@/components/tables/verdict-table";
import { VerdictRadarChart } from "@/components/charts/radar-chart";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { useVerdict } from "@/hooks/use-verdict";
import { DEFAULT_US_TICKERS, NASDAQ_50_TICKERS } from "@/lib/constants";
import { Scale } from "lucide-react";

export default function USVerdictPage() {
  const [tickers, setTickers] = useState<string[]>(DEFAULT_US_TICKERS);
  const [verdictConfig, setVerdictConfig] = useState<Record<string, unknown>>({});
  const { run, isRunning, results, error } = useVerdict("US");

  const handleRun = () => {
    if (tickers.length === 0) return;
    run({
      tickers,
      market: "US",
      date_range: ["", ""],
      skip_layers: [],
      weights: { core: 0.3, strategy: 0.3, ml_features: 0.2, robustness: 0.2 },
    });
  };

  const avgScore = results.length > 0
    ? results.reduce((a, r) => a + r.weighted_score, 0) / results.length
    : 0;

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NASDAQ_50_TICKERS} market="US" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold">Verdict Engine</h3>
          <TickerInput defaultTickers={DEFAULT_US_TICKERS} onTickersChange={setTickers} />
          <VerdictConfig onConfigChange={(c) => setVerdictConfig(c)} />
          <Button className="w-full" onClick={handleRun} disabled={isRunning}>
            {isRunning ? "Computing…" : <><Scale className="mr-1 h-4 w-4" /> Run Verdict</>}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <div className="md:col-span-3 space-y-4">
          {isRunning && <Spinner />}
          {results.length > 0 && (
            <>
              <MetricsGrid>
                <MetricCard label="Stocks Analyzed" value={results.length} />
                <MetricCard label="Avg Weighted Score" value={avgScore.toFixed(2)} />
                <MetricCard
                  label="Buy Signals"
                  value={results.filter((r) => r.verdict === "BUY" || r.verdict === "STRONG_BUY").length}
                  color="text-green-500"
                />
                <MetricCard
                  label="Sell Signals"
                  value={results.filter((r) => r.verdict === "SELL" || r.verdict === "STRONG_SELL").length}
                  color="text-red-500"
                />
              </MetricsGrid>

              {/* Radar chart for first result */}
              {results[0] && (
                <div className="content-panel p-4">
                  <h4 className="text-sm font-semibold mb-2">{results[0].ticker} — Layer Scores</h4>
                  <VerdictRadarChart
                    data={[
                      { layer: "Core", score: results[0].core_score },
                      { layer: "Strategy", score: results[0].strategy_score },
                      { layer: "ML", score: results[0].ml_score },
                      { layer: "Robustness", score: results[0].robustness_score },
                    ]}
                  />
                </div>
              )}

              <div className="content-panel p-4">
                <VerdictTable data={results} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
