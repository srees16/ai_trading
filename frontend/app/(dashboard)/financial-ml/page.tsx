"use client";

import { useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { Spinner } from "@/components/common/spinner";
import { useFmlChapters, useFmlRun } from "@/hooks/use-fml";
import { FML_CATEGORIES, DEFAULT_FML_TICKERS } from "@/lib/constants";
import { format, parse } from "date-fns";
import { Play, FlaskConical, CalendarDays, Tag, Upload, ChevronDown, ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";

type TickerMode = "default" | "manual" | "csv";

export default function FinancialMLPage() {
  const chaptersQ = useFmlChapters();
  const { run, isRunning, progress, error, abort, isAborted } = useFmlRun();
  const [selected, setSelected] = useState<string[]>([]);

  // Ticker state
  const [tickerMode, setTickerMode] = useState<TickerMode>("default");
  const [manualTickers, setManualTickers] = useState("MSFT, GOOG, NVDA, AMD");
  const [csvTickers, setCsvTickers] = useState<string[]>([]);

  // Collapsed results state
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  // Date state
  const [dateStart, setDateStart] = useState<Date>(new Date(2020, 0, 1));
  const [dateEnd, setDateEnd] = useState<Date>(new Date(2024, 11, 31));
  const [startOpen, setStartOpen] = useState(false);
  const [endOpen, setEndOpen] = useState(false);

  const chapters = chaptersQ.data ?? [];
  const grouped = FML_CATEGORIES.map((cat) => ({
    category: cat,
    items: chapters.filter((c) => c.category === cat),
  })).filter((g) => g.items.length > 0);

  const toggleChapter = (key: string, checked: boolean) => {
    setSelected((prev) => (checked ? [...prev, key] : prev.filter((k) => k !== key)));
  };

  const getActiveTickers = (): string[] => {
    if (tickerMode === "default") return DEFAULT_FML_TICKERS;
    if (tickerMode === "csv") return csvTickers;
    return manualTickers
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
  };

  const handleCsvUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      const parsed = text
        .split(/[\n,]/)
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t && t !== "TICKER" && /^[A-Z]{1,10}$/.test(t));
      setCsvTickers(parsed);
    };
    reader.readAsText(file);
  };

  const handleRun = () => {
    if (selected.length === 0) return;
    const tickers = getActiveTickers();
    run({
      chapters: selected,
      tickers: tickers.length > 0 ? tickers : undefined,
      date_start: format(dateStart, "yyyy-MM-dd"),
      date_end: format(dateEnd, "yyyy-MM-dd"),
    });
  };

  const pct = progress ? Math.round((progress.completed / progress.total) * 100) : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <FlaskConical className="h-5 w-5 text-purple-500" />
        <h2 className="text-lg font-semibold">Advanced Financial Machine Learning</h2>
        <span className="text-xs text-muted-foreground">Based on Marcos López de Prado</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* ─── Left panel: inputs + chapter selection ─── */}
        <div className="content-panel p-4 space-y-5">
          {/* Ticker selection */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Tag className="h-3.5 w-3.5 text-muted-foreground" />
              <h3 className="text-sm font-semibold">Select Stocks</h3>
            </div>
            <div className="flex gap-1">
              {(["default", "manual", "csv"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setTickerMode(mode)}
                  className={`flex-1 px-2 py-1 text-xs rounded border transition-colors ${
                    tickerMode === mode
                      ? "bg-primary/10 border-primary text-primary font-medium"
                      : "border-border text-muted-foreground hover:bg-muted"
                  }`}
                >
                  {mode === "default" ? "Default" : mode === "manual" ? "Manual" : "CSV"}
                </button>
              ))}
            </div>

            {tickerMode === "default" && (
              <div className="text-xs text-muted-foreground bg-muted/50 rounded px-2 py-1.5">
                {DEFAULT_FML_TICKERS.join(", ")}
              </div>
            )}
            {tickerMode === "manual" && (
              <Input
                placeholder="MSFT, GOOG, NVDA, AMD"
                value={manualTickers}
                onChange={(e) => setManualTickers(e.target.value)}
                className="text-xs"
              />
            )}
            {tickerMode === "csv" && (
              <div className="space-y-1">
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer border border-dashed rounded px-2 py-1.5 hover:bg-muted/50">
                  <Upload className="h-3.5 w-3.5" />
                  {csvTickers.length > 0
                    ? `${csvTickers.length} tickers loaded`
                    : "Upload CSV"}
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleCsvUpload}
                    className="hidden"
                  />
                </label>
                {csvTickers.length > 0 && (
                  <div className="text-xs text-muted-foreground bg-muted/50 rounded px-2 py-1">
                    {csvTickers.slice(0, 8).join(", ")}
                    {csvTickers.length > 8 && ` +${csvTickers.length - 8} more`}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Date range */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <CalendarDays className="h-3.5 w-3.5 text-muted-foreground" />
              <h3 className="text-sm font-semibold">Date Range</h3>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label className="text-xs text-muted-foreground">Start</Label>
                <Popover open={startOpen} onOpenChange={setStartOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn("w-full justify-start text-left text-xs font-normal", !dateStart && "text-muted-foreground")}
                    >
                      <CalendarDays className="mr-1.5 h-3.5 w-3.5" />
                      {format(dateStart, "MMM d, yyyy")}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      captionLayout="dropdown"
                      selected={dateStart}
                      onSelect={(d) => { if (d) { setDateStart(d); setStartOpen(false); } }}
                      defaultMonth={dateStart}
                      startMonth={new Date(2020, 0)}
                      endMonth={new Date(2030, 11)}
                    />
                  </PopoverContent>
                </Popover>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">End</Label>
                <Popover open={endOpen} onOpenChange={setEndOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn("w-full justify-start text-left text-xs font-normal", !dateEnd && "text-muted-foreground")}
                    >
                      <CalendarDays className="mr-1.5 h-3.5 w-3.5" />
                      {format(dateEnd, "MMM d, yyyy")}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      captionLayout="dropdown"
                      selected={dateEnd}
                      onSelect={(d) => { if (d) { setDateEnd(d); setEndOpen(false); } }}
                      defaultMonth={dateEnd}
                      startMonth={new Date(2020, 0)}
                      endMonth={new Date(2030, 11)}
                    />
                  </PopoverContent>
                </Popover>
              </div>
            </div>
            <p className="text-[0.65rem] text-muted-foreground">
              Applies to chapters that fetch live market data. Synthetic data chapters are unaffected.
            </p>
          </div>

          {/* Chapter selection */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">Select Chapters</h3>

            <div className="flex gap-2">
              <Button className="flex-1" onClick={handleRun} disabled={isRunning || selected.length === 0}>
                {isRunning ? "Running…" : <><Play className="mr-1 h-4 w-4" /> Run Analyses ({selected.length})</>}
              </Button>
              {isRunning && (
                <Button variant="destructive" onClick={abort}>
                  <X className="mr-1 h-4 w-4" /> Cancel
                </Button>
              )}
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}

            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSelected(chapters.map((c) => c.key))}>All</Button>
              <Button size="sm" variant="outline" onClick={() => setSelected([])}>None</Button>
            </div>

            {chaptersQ.isLoading ? (
              <Spinner />
            ) : (
              <div className="space-y-3 max-h-[400px] overflow-y-auto">
                {grouped.map((g) => (
                  <div key={g.category}>
                    <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">{g.category}</p>
                    {g.items.map((ch) => (
                      <div key={ch.key} className="flex items-start gap-2 py-0.5">
                        <Checkbox
                          checked={selected.includes(ch.key)}
                          onCheckedChange={(v) => toggleChapter(ch.key, v === true)}
                          className="mt-0.5"
                        />
                        <div className="flex-1 min-w-0">
                          <Label className="text-sm font-normal">{ch.title}</Label>
                          {ch.description && (
                            <p className="text-[0.65rem] text-muted-foreground leading-tight">{ch.description}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ─── Right panel: results ─── */}
        <div className="md:col-span-3 space-y-4">
          {isRunning && progress && (
            <div className="content-panel p-4">
              <div className="flex items-center gap-3">
                <Spinner />
                <div>
                  <p className="text-sm font-medium">Running chapters… {progress.completed}/{progress.total}</p>
                  <p className="text-xs text-muted-foreground">{pct}% complete</p>
                </div>
              </div>
            </div>
          )}

          {isAborted && progress && (
            <div className="content-panel p-4 border-destructive">
              <p className="text-sm font-medium text-destructive">
                Aborted — {progress.completed}/{progress.total} chapters completed
              </p>
            </div>
          )}

          {progress?.chapters &&
            Object.entries(progress.chapters).map(([key, ch]) => {
              const isCollapsed = collapsed.has(key);
              const canCollapse = ch.status === "done" || ch.status === "error";
              return (
                <div key={key} className="content-panel p-4 space-y-2">
                  <button
                    className="flex items-center gap-2 w-full text-left"
                    onClick={() => {
                      if (!canCollapse) return;
                      setCollapsed((prev) => {
                        const next = new Set(prev);
                        next.has(key) ? next.delete(key) : next.add(key);
                        return next;
                      });
                    }}
                  >
                    {canCollapse && (
                      isCollapsed
                        ? <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                        : <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                    )}
                    <h4 className="text-sm font-semibold">{key}</h4>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      ch.status === "done" ? "bg-green-500/10 text-green-600" :
                      ch.status === "running" ? "bg-blue-500/10 text-blue-600" :
                      ch.status === "error" ? "bg-red-500/10 text-red-600" :
                      "bg-muted text-muted-foreground"
                    }`}>
                      {ch.status}
                    </span>
                  </button>

                  {!isCollapsed && (
                    <>
                      {ch.text_output && <pre className="text-xs bg-muted/50 rounded p-3 overflow-auto max-h-[80vh] whitespace-pre-wrap">{ch.text_output}</pre>}
                      {ch.error_message && <p className="text-sm text-destructive">{ch.error_message}</p>}

                      {ch.figures.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {ch.figures.map((fig, i) => (
                            <img key={i} src={`data:image/png;base64,${fig}`} alt={`${key} figure ${i + 1}`} className="rounded border" />
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
