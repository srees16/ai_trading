"use client";

import { useState, useCallback } from "react";
import { usStocksApi, type BacktestRequest, type BacktestResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, Play, BookOpen, RefreshCw, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";

interface ChapterDef {
  key: string;
  title: string;
  description: string;
  needsTickers: boolean;
}

const CHAPTER_GROUPS: { name: string; chapters: ChapterDef[] }[] = [
  {
    name: "Data Structures",
    chapters: [
      { key: "ch02", title: "Ch.2 — Financial Data Structures", description: "Tick, volume, dollar bars and information-driven bars", needsTickers: true },
      { key: "ch03", title: "Ch.3 — Labeling (Triple-Barrier)", description: "Triple-barrier labeling method for supervised learning", needsTickers: true },
      { key: "ch04", title: "Ch.4 — Sample Weights", description: "Sample uniqueness, weights, and sequential bootstrap", needsTickers: true },
    ],
  },
  {
    name: "Features",
    chapters: [
      { key: "ch05", title: "Ch.5 — Fractional Differentiation", description: "Memory-preserving stationarity transforms", needsTickers: true },
      { key: "ch08", title: "Ch.8 — Feature Importance", description: "MDI, MDA, and SFI feature importance methods", needsTickers: true },
      { key: "ch17", title: "Ch.17 — Structural Breaks", description: "CUSUM, Chow, and Supremum ADF tests", needsTickers: true },
      { key: "ch18", title: "Ch.18 — Entropy Features", description: "Shannon, plug-in, Lempel-Ziv, and Kontoyiannis entropy", needsTickers: true },
      { key: "ch19", title: "Ch.19 — Microstructural Features", description: "Tick rule, VPIN, Kyle lambda, Amihud lambda", needsTickers: true },
    ],
  },
  {
    name: "Modeling",
    chapters: [
      { key: "ch06", title: "Ch.6 — Ensemble Methods", description: "Bagging, random forest, and boosting with purging", needsTickers: true },
      { key: "ch07", title: "Ch.7 — Cross-Validation", description: "Purged K-Fold and combinatorial purged CV", needsTickers: true },
      { key: "ch09", title: "Ch.9 — Hyper-Parameter Tuning", description: "Grid search with purged CV and scoring", needsTickers: true },
      { key: "ch10", title: "Ch.10 — Bet Sizing", description: "Optimal bet sizing from classifier probabilities", needsTickers: true },
    ],
  },
  {
    name: "Backtesting",
    chapters: [
      { key: "ch11", title: "Ch.11 — Dangers of Backtesting", description: "Overfitting and selection bias in backtests", needsTickers: false },
      { key: "ch13", title: "Ch.13 — Synthetic Backtesting", description: "Synthetic data generation for backtests", needsTickers: true },
      { key: "ch14", title: "Ch.14 — Backtest Statistics", description: "Deflated Sharpe, probabilistic Sharpe ratio", needsTickers: true },
      { key: "ch15", title: "Ch.15 — Strategy Risk", description: "Drawdown analysis and risk of ruin", needsTickers: true },
    ],
  },
  {
    name: "Portfolio",
    chapters: [
      { key: "ch16", title: "Ch.16 — ML Asset Allocation", description: "Hierarchical risk parity (HRP) and CLA", needsTickers: true },
    ],
  },
  {
    name: "Computation",
    chapters: [
      { key: "ch20", title: "Ch.20 — Multiprocessing", description: "Parallel execution with multiprocessing engine", needsTickers: false },
      { key: "ch21", title: "Ch.21 — Brute Force & Quantum", description: "Combinatorial purged cross-validation", needsTickers: false },
    ],
  },
];

interface ChapterResult {
  text?: string;
  figures?: string[];  // base64 PNG images
  tables?: Record<string, unknown>[];
  error?: string;
}

export default function FinanceMLPage() {
  const [tickers, setTickers] = useState("MSFT, GOOG, NVDA, AMD");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [running, setRunning] = useState(false);
  const [runningChapter, setRunningChapter] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<Record<string, ChapterResult>>({});
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set());

  const allChapters = CHAPTER_GROUPS.flatMap((g) => g.chapters);

  const parseTickerList = () => tickers.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean);

  const runChapter = async (ch: ChapterDef) => {
    const tickerList = parseTickerList();
    if (ch.needsTickers && tickerList.length === 0) { toast.error("Enter tickers first"); return; }
    setRunningChapter(ch.key);
    try {
      const res = await usStocksApi.backtest({
        strategy_id: `fml_${ch.key}`,
        tickers: tickerList,
        start_date: startDate,
        end_date: endDate,
        parameters: { chapter: ch.key },
      });
      const chResult: ChapterResult = {};
      if (res.charts?.length) chResult.figures = res.charts.map((c: any) => c.image || "").filter(Boolean);
      if (res.tables?.length) chResult.tables = res.tables;
      if (res.metadata) chResult.text = typeof res.metadata === "string" ? res.metadata : JSON.stringify(res.metadata, null, 2);
      setResults((prev) => ({ ...prev, [ch.key]: chResult }));
      setExpandedChapters((prev) => new Set(prev).add(ch.key));
      toast.success(`${ch.title} complete`);
    } catch (err: any) {
      setResults((prev) => ({ ...prev, [ch.key]: { error: err?.message || "Failed" } }));
      toast.error(`${ch.key} failed`);
    } finally { setRunningChapter(null); }
  };

  const runAll = async () => {
    setRunning(true);
    setProgress(0);
    const tickerList = parseTickerList();
    for (let i = 0; i < allChapters.length; i++) {
      const ch = allChapters[i];
      setProgress(Math.round(((i) / allChapters.length) * 100));
      if (ch.needsTickers && tickerList.length === 0) continue;
      try {
        const res = await usStocksApi.backtest({
          strategy_id: `fml_${ch.key}`,
          tickers: tickerList,
          start_date: startDate,
          end_date: endDate,
          parameters: { chapter: ch.key },
        });
        const chResult: ChapterResult = {};
        if (res.charts?.length) chResult.figures = res.charts.map((c: any) => c.image || "").filter(Boolean);
        if (res.tables?.length) chResult.tables = res.tables;
        if (res.metadata) chResult.text = typeof res.metadata === "string" ? res.metadata : JSON.stringify(res.metadata, null, 2);
        setResults((prev) => ({ ...prev, [ch.key]: chResult }));
      } catch (err: any) {
        setResults((prev) => ({ ...prev, [ch.key]: { error: err?.message || "Failed" } }));
      }
    }
    setProgress(100);
    setRunning(false);
    toast.success("All analyses complete");
  };

  const toggleChapter = (key: string) => {
    setExpandedChapters((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Advanced Financial Machine Learning</h1>
        <p className="text-sm text-muted-foreground">19 ML techniques from Marcos López de Prado&apos;s methodology</p>
      </div>

      {/* Config */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">Ticker Selection</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div><Label className="text-xs">Tickers (comma-separated)</Label><Input value={tickers} onChange={(e) => setTickers(e.target.value)} /></div>
            <div className="grid grid-cols-2 gap-3">
              <div><Label className="text-xs">Start Date</Label><Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></div>
              <div><Label className="text-xs">End Date</Label><Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Run</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Button onClick={runAll} disabled={running} className="w-full" size="lg">
              {running ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {running ? "Running All..." : "Run All Analyses"}
            </Button>
            {running && <Progress value={progress} />}
            <p className="text-xs text-center text-muted-foreground">
              {Object.keys(results).length} / {allChapters.length} completed
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Chapter Groups */}
      <Tabs defaultValue={CHAPTER_GROUPS[0].name}>
        <TabsList className="flex-wrap">
          {CHAPTER_GROUPS.map((g) => (
            <TabsTrigger key={g.name} value={g.name}>{g.name}</TabsTrigger>
          ))}
        </TabsList>
        {CHAPTER_GROUPS.map((group) => (
          <TabsContent key={group.name} value={group.name} className="space-y-3">
            {group.chapters.map((ch) => (
              <Card key={ch.key}>
                <CardHeader className="cursor-pointer pb-3" onClick={() => toggleChapter(ch.key)}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {expandedChapters.has(ch.key) ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      <div>
                        <CardTitle className="text-sm">{ch.title}</CardTitle>
                        <CardDescription className="text-xs">{ch.description}</CardDescription>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {results[ch.key] && !results[ch.key].error && <Badge className="bg-profit/20 text-profit">Done</Badge>}
                      {results[ch.key]?.error && <Badge variant="destructive">Error</Badge>}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e: React.MouseEvent) => { e.stopPropagation(); runChapter(ch); }}
                        disabled={running || runningChapter === ch.key}
                      >
                        {runningChapter === ch.key ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                {expandedChapters.has(ch.key) && results[ch.key] && (
                  <CardContent className="space-y-4 border-t pt-4">
                    {results[ch.key].error && (
                      <p className="text-sm text-loss">{results[ch.key].error}</p>
                    )}
                    {results[ch.key].figures?.map((fig, i) => (
                      <img key={i} src={`data:image/png;base64,${fig}`} alt={`${ch.key} figure ${i + 1}`} className="w-full rounded border" />
                    ))}
                    {results[ch.key].tables?.map((tbl, i) => (
                      <div key={i} className="overflow-auto">
                        <pre className="rounded bg-muted p-3 text-xs">{JSON.stringify(tbl, null, 2)}</pre>
                      </div>
                    ))}
                    {results[ch.key].text && (
                      <pre className="rounded bg-muted p-3 text-xs whitespace-pre-wrap">{results[ch.key].text}</pre>
                    )}
                  </CardContent>
                )}
              </Card>
            ))}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
