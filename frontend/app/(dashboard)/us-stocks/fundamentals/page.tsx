"use client";

import { useState } from "react";
import { TickerInput } from "@/components/forms/ticker-input";
import { FundamentalTable, buildFundamentalRows } from "@/components/tables/fundamental-table";
import { FundamentalCharts } from "@/components/charts/fundamental-charts";
import { ScoreInterpretations } from "@/components/common/score-interpretations";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { Button } from "@/components/ui/button";
import { useAnalysis } from "@/hooks/use-analysis";
import { DEFAULT_US_TICKERS, NASDAQ_50_TICKERS } from "@/lib/constants";
import { BarChart3 } from "lucide-react";

export default function USFundamentalsPage() {
  const [tickers, setTickers] = useState<string[]>(DEFAULT_US_TICKERS);
  const { run, isRunning, result, error } = useAnalysis("US");

  const handleRun = () => {
    if (tickers.length > 0) run({ tickers, market: "US", period: "1y" });
  };

  const signals = result?.signals ?? [];
  const rows = buildFundamentalRows(signals);

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NASDAQ_50_TICKERS} market="US" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold">Fundamental Analysis</h3>
          <TickerInput defaultTickers={DEFAULT_US_TICKERS} onTickersChange={setTickers} />
          <Button className="w-full" onClick={handleRun} disabled={isRunning}>
            {isRunning ? "Loading…" : <><BarChart3 className="mr-1 h-4 w-4" /> Analyze</>}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <ScoreInterpretations />
        </div>

        <div className="md:col-span-3 space-y-4">
          {isRunning && <Spinner />}
          {rows.length > 0 && (
            <>
              <div className="content-panel p-4">
                <FundamentalCharts data={rows} />
              </div>
              <div className="content-panel p-4">
                <FundamentalTable data={rows} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
