"use client";

import { useState } from "react";
import { usStocksApi, type TradingSignal } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, getDecisionBadgeClass } from "@/lib/utils";
import { Loader2, Play, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { toast } from "sonner";

interface VerdictResult {
  ticker: string;
  score: number;
  classification: string;
  confidence: number;
  rsi: number | null;
  current_price: number | null;
  reasoning: string;
}

export default function USVerdictPage() {
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL, TSLA, NVDA");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<VerdictResult[]>([]);

  const handleRun = async () => {
    const tickerList = tickers.split(/[,\n]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickerList.length === 0) return;

    setLoading(true);
    try {
      const data = await usStocksApi.analyze(tickerList);
      if (data.success && data.signals) {
        // Deduplicate by ticker — take highest abs score
        const byTicker = new Map<string, VerdictResult>();
        for (const s of data.signals) {
          const mapped: VerdictResult = {
            ticker: s.ticker,
            score: s.decision_score,
            classification: s.decision,
            confidence: s.sentiment_confidence || 0,
            rsi: s.rsi,
            current_price: s.current_price,
            reasoning: s.reasoning,
          };
          const existing = byTicker.get(mapped.ticker);
          if (!existing || Math.abs(mapped.score) > Math.abs(existing.score)) {
            byTicker.set(mapped.ticker, mapped);
          }
        }
        setResults(Array.from(byTicker.values()));
        toast.success(`Verdict complete for ${byTicker.size} tickers`);
      }
    } catch {
      toast.error("Verdict failed");
    } finally {
      setLoading(false);
    }
  };

  const buyCount = results.filter((r) => r.classification?.toUpperCase().includes("BUY")).length;
  const sellCount = results.filter((r) => r.classification?.toUpperCase().includes("SELL")).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Integrated Verdict</h1>
        <p className="text-sm text-muted-foreground">
          Multi-layer analysis: Core + Strategy + ML Features + Robustness
        </p>
      </div>

      <Card>
        <CardContent className="flex items-end gap-4 pt-6">
          <div className="flex-1 space-y-2">
            <Label>Tickers</Label>
            <Input value={tickers} onChange={(e) => setTickers(e.target.value)} placeholder="AAPL, MSFT..." />
          </div>
          <Button onClick={handleRun} disabled={loading}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run Verdict
          </Button>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCard title="Total" value={String(results.length)} icon={<BarChart3 className="h-4 w-4" />} />
            <MetricCard title="Buy" value={String(buyCount)} changeType="positive" icon={<TrendingUp className="h-4 w-4" />} />
            <MetricCard title="Sell" value={String(sellCount)} changeType="negative" icon={<TrendingDown className="h-4 w-4" />} />
            <MetricCard title="Hold" value={String(results.length - buyCount - sellCount)} />
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {results.map((r) => (
              <Card key={r.ticker}>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="font-mono text-lg">{r.ticker}</CardTitle>
                    <Badge className={getDecisionBadgeClass(r.classification)}>{r.classification}</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Score</span>
                      <span className="font-mono font-bold">{r.score.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Confidence</span>
                      <span className="font-mono">{(r.confidence * 100).toFixed(1)}%</span>
                    </div>
                    {r.current_price != null && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Price</span>
                        <span className="font-mono">${formatNumber(r.current_price)}</span>
                      </div>
                    )}
                    {r.rsi != null && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">RSI</span>
                        <span className="font-mono">{formatNumber(r.rsi, 1)}</span>
                      </div>
                    )}
                    {/* Score bar */}
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary transition-all"
                        style={{ width: `${Math.min(Math.abs(r.score) * 100, 100)}%` }}
                      />
                    </div>
                    {r.reasoning && (
                      <p className="mt-2 text-xs text-muted-foreground line-clamp-2">{r.reasoning}</p>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
