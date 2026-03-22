"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { NIFTY_50_TICKERS } from "@/lib/constants";
import { useSignalHistory, useBacktestHistory } from "@/hooks/use-history";
import { formatPct, formatCurrency } from "@/lib/utils";
import { DecisionBadge } from "@/components/common/decision-badge";

export default function INDHistoryPage() {
  const [tab, setTab] = useState("signals");
  const signalQ = useSignalHistory("IND");
  const backtestQ = useBacktestHistory("IND");

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="signals">Signal History</TabsTrigger>
          <TabsTrigger value="backtests">Backtest History</TabsTrigger>
        </TabsList>

        <TabsContent value="signals" className="mt-4">
          {signalQ.isLoading && <Spinner />}
          {signalQ.data && (
            <div className="content-panel p-4">
              <p className="text-xs text-muted-foreground mb-2">{signalQ.data.total} total signals</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b text-muted-foreground text-left">
                    <th className="py-2 pr-3">Date</th><th className="py-2 pr-3">Ticker</th><th className="py-2 pr-3">Decision</th><th className="py-2 pr-3">Score</th><th className="py-2 pr-3">Sentiment</th><th className="py-2 pr-3">Price</th><th className="py-2">RSI</th>
                  </tr></thead>
                  <tbody>{signalQ.data.data.map((r) => (
                    <tr key={r.id} className="border-b">
                      <td className="py-2 pr-3 text-xs">{new Date(r.created_at).toLocaleString()}</td>
                      <td className="py-2 pr-3 font-mono font-medium">{r.ticker}</td>
                      <td className="py-2 pr-3"><DecisionBadge decision={r.decision} /></td>
                      <td className="py-2 pr-3">{r.decision_score.toFixed(2)}</td>
                      <td className="py-2 pr-3">{r.sentiment_label}</td>
                      <td className="py-2 pr-3">{formatCurrency(r.current_price)}</td>
                      <td className="py-2">{r.rsi.toFixed(1)}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="backtests" className="mt-4">
          {backtestQ.isLoading && <Spinner />}
          {backtestQ.data && (
            <div className="content-panel p-4">
              <p className="text-xs text-muted-foreground mb-2">{backtestQ.data.total} total backtests</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b text-muted-foreground text-left">
                    <th className="py-2 pr-3">Date</th><th className="py-2 pr-3">Strategy</th><th className="py-2 pr-3">Return</th><th className="py-2 pr-3">Sharpe</th><th className="py-2 pr-3">Max DD</th><th className="py-2">Trades</th>
                  </tr></thead>
                  <tbody>{backtestQ.data.data.map((r) => (
                    <tr key={r.id} className="border-b">
                      <td className="py-2 pr-3 text-xs">{new Date(r.created_at).toLocaleString()}</td>
                      <td className="py-2 pr-3">{r.strategy_name}</td>
                      <td className={`py-2 pr-3 ${r.total_return >= 0 ? "pnl-positive" : "pnl-negative"}`}>{formatPct(r.total_return)}</td>
                      <td className="py-2 pr-3">{r.sharpe_ratio.toFixed(2)}</td>
                      <td className="py-2 pr-3 text-red-500">{formatPct(r.max_drawdown)}</td>
                      <td className="py-2">{r.total_trades}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
