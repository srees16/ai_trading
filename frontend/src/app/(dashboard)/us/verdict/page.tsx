"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Loader2, Play, Shield } from "lucide-react";
import { toast } from "sonner";

// Verdict page calls the backend IntegratedScorer via API
// The IntegratedScorer runs 4 layers: core, strategy, ml_features, robustness

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9001";

interface VerdictResult {
  ticker: string;
  score: number;
  classification: string;
  confidence: number;
  layers?: Record<string, number>;
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
      const res = await fetch(`${API_BASE}/us-stocks/analysis`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: tickerList }),
      });
      const data = await res.json();
      if (data.success && data.signals) {
        const mapped = data.signals.map((s: any) => ({
          ticker: s.ticker,
          score: s.decision_score,
          classification: s.decision,
          confidence: s.sentiment_confidence || 0,
        }));
        // Deduplicate by ticker — take highest abs score
        const byTicker = new Map<string, VerdictResult>();
        for (const r of mapped) {
          const existing = byTicker.get(r.ticker);
          if (!existing || Math.abs(r.score) > Math.abs(existing.score)) {
            byTicker.set(r.ticker, r);
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

  const getClassColor = (cls: string) => {
    const c = cls?.toUpperCase() || "";
    if (c.includes("STRONG_BUY") || c.includes("STRONG BUY")) return "badge-strong-buy";
    if (c.includes("BUY")) return "badge-buy";
    if (c.includes("HOLD")) return "badge-hold";
    if (c.includes("STRONG_SELL") || c.includes("STRONG SELL")) return "badge-strong-sell";
    if (c.includes("SELL")) return "badge-sell";
    return "";
  };

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
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {results.map((r) => (
            <Card key={r.ticker}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="font-mono text-lg">{r.ticker}</CardTitle>
                  <Badge className={getClassColor(r.classification)}>{r.classification}</Badge>
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
                  {/* Score bar */}
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary transition-all"
                      style={{ width: `${Math.min(Math.abs(r.score) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
