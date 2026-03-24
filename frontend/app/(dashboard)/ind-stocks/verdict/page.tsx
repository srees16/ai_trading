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
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useVerdict } from "@/hooks/use-verdict";
import { DEFAULT_IND_TICKERS, NIFTY_50_TICKERS } from "@/lib/constants";
import { Scale } from "lucide-react";

export default function INDVerdictPage() {
  const [tickers, setTickers] = useState<string[]>(DEFAULT_IND_TICKERS);
  const [verdictConfig, setVerdictConfig] = useState<Record<string, unknown>>({});
  const { run, isRunning, results, error } = useVerdict("IND");

  const handleRun = () => {
    if (tickers.length === 0) return;
    run({
      tickers,
      market: "IND",
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
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold">Verdict Engine</h3>
          <TickerInput defaultTickers={DEFAULT_IND_TICKERS} onTickersChange={setTickers} />
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

              {results.length > 0 && (
                <div className="content-panel p-4">
                  <Tabs defaultValue={results[0].ticker}>
                    <TabsList className="flex flex-wrap gap-1 h-auto">
                      {results.map((r) => (
                        <TabsTrigger key={r.ticker} value={r.ticker}>
                          {r.ticker}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                    {results.map((r) => (
                      <TabsContent key={r.ticker} value={r.ticker}>
                        <h4 className="text-sm font-semibold mb-2">{r.ticker} — Layer Scores</h4>
                        <VerdictRadarChart
                          data={[
                            { layer: "Core", score: r.core_score },
                            { layer: "Strategy", score: r.strategy_score },
                            { layer: "ML", score: r.ml_score },
                            { layer: "Robustness", score: r.robustness_score },
                          ]}
                        />
                      </TabsContent>
                    ))}
                  </Tabs>
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
