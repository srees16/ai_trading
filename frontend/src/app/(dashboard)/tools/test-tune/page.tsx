"use client";

import { useState } from "react";
import { usStocksApi, type BacktestResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, Play, RefreshCw, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";

interface ChapterDef {
  key: string;
  title: string;
  description: string;
}

const CHAPTER_GROUPS: { name: string; chapters: ChapterDef[] }[] = [
  {
    name: "Foundations",
    chapters: [
      { key: "ch01", title: "Ch.1 — Introduction", description: "Overview of trading system testing methodology" },
      { key: "ch02", title: "Ch.2 — Pre-Optimization Issues", description: "Data snooping, look-ahead bias, survivorship bias" },
    ],
  },
  {
    name: "Optimization",
    chapters: [
      { key: "ch03", title: "Ch.3 — Optimization Issues", description: "Curve fitting, over-optimization detection" },
      { key: "ch04", title: "Ch.4 — Post-Optimization Issues", description: "Walk-forward analysis and robustness testing" },
    ],
  },
  {
    name: "Performance Estimation",
    chapters: [
      { key: "ch05", title: "Ch.5 — Unbiased Performance Estimation", description: "Deflated Sharpe ratio, multiple testing correction" },
      { key: "ch06", title: "Ch.6 — Trade-Based Analysis", description: "Trade-level statistics and profit factor analysis" },
    ],
  },
  {
    name: "Statistical Testing",
    chapters: [
      { key: "ch07", title: "Ch.7 — Permutation Tests", description: "Monte Carlo permutation tests for strategy validation" },
    ],
  },
];

interface ChapterResult {
  text?: string;
  figures?: string[];
  tables?: Record<string, unknown>[];
  error?: string;
}

export default function TestTunePage() {
  const [tickers, setTickers] = useState("SPY, QQQ, IWM, DIA");
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
    if (tickerList.length === 0) { toast.error("Enter tickers first"); return; }
    setRunningChapter(ch.key);
    try {
      const res = await usStocksApi.backtest({
        strategy_id: `tts_${ch.key}`,
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
      setProgress(Math.round(((i) / allChapters.length) * 100));
      try {
        const res = await usStocksApi.backtest({
          strategy_id: `tts_${allChapters[i].key}`,
          tickers: tickerList,
          start_date: startDate,
          end_date: endDate,
          parameters: { chapter: allChapters[i].key },
        });
        const chResult: ChapterResult = {};
        if (res.charts?.length) chResult.figures = res.charts.map((c: any) => c.image || "").filter(Boolean);
        if (res.tables?.length) chResult.tables = res.tables;
        if (res.metadata) chResult.text = typeof res.metadata === "string" ? res.metadata : JSON.stringify(res.metadata, null, 2);
        setResults((prev) => ({ ...prev, [allChapters[i].key]: chResult }));
      } catch (err: any) {
        setResults((prev) => ({ ...prev, [allChapters[i].key]: { error: err?.message || "Failed" } }));
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
        <h1 className="text-2xl font-bold">Testing & Tuning Market Trading Systems</h1>
        <p className="text-sm text-muted-foreground">7 chapters on statistical validation of trading strategies</p>
      </div>

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
                      <Button size="sm" variant="outline" onClick={(e: React.MouseEvent) => { e.stopPropagation(); runChapter(ch); }} disabled={running || runningChapter === ch.key}>
                        {runningChapter === ch.key ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                {expandedChapters.has(ch.key) && results[ch.key] && (
                  <CardContent className="space-y-4 border-t pt-4">
                    {results[ch.key].error && <p className="text-sm text-loss">{results[ch.key].error}</p>}
                    {results[ch.key].figures?.map((fig, i) => (
                      <img key={i} src={`data:image/png;base64,${fig}`} alt={`${ch.key} fig ${i + 1}`} className="w-full rounded border" />
                    ))}
                    {results[ch.key].tables?.map((tbl, i) => (
                      <pre key={i} className="rounded bg-muted p-3 text-xs overflow-auto">{JSON.stringify(tbl, null, 2)}</pre>
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
