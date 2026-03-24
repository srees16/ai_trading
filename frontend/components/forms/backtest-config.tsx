"use client";

import { useState } from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { PERIOD_OPTIONS } from "@/lib/constants";
import type { StrategyInfo, StrategyParam } from "@/lib/types";

interface BacktestConfigProps {
  strategies: StrategyInfo[];
  categories: string[];
  onConfigChange: (config: {
    strategy_id: string;
    params: Record<string, number | boolean>;
    tickers: string;
    initial_capital: number;
    period: string;
    start_date: string;
    end_date: string;
  }) => void;
  defaultTickers?: string;
}

export function BacktestConfig({ strategies, categories, onConfigChange, defaultTickers = "" }: BacktestConfigProps) {
  const [category, setCategory] = useState("All");
  const [strategyId, setStrategyId] = useState(strategies[0]?.id || "");
  const [params, setParams] = useState<Record<string, number | boolean>>({});
  const [tickers, setTickers] = useState(defaultTickers);
  const [capital, setCapital] = useState(100000);
  const [period, setPeriod] = useState("1y");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const filtered = category === "All" ? strategies : strategies.filter((s) => s.category === category);
  const selected = strategies.find((s) => s.id === strategyId);

  const emitChange = (overrides?: Partial<Record<string, unknown>>) => {
    onConfigChange({
      strategy_id: strategyId,
      params,
      tickers,
      initial_capital: capital,
      period,
      start_date: startDate,
      end_date: endDate,
      ...overrides,
    });
  };

  const handleParamChange = (name: string, value: number | boolean) => {
    const next = { ...params, [name]: value };
    setParams(next);
    emitChange({ params: next });
  };

  return (
    <div className="space-y-4">
      {/* Category */}
      <div>
        <Label className="text-xs">Strategy Category</Label>
        <Select value={category} onValueChange={setCategory}>
          <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All</SelectItem>
            {categories.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {/* Strategy */}
      <div>
        <Label className="text-xs">Select Strategy</Label>
        <Select value={strategyId} onValueChange={(v) => { setStrategyId(v); setParams({}); emitChange({ strategy_id: v, params: {} }); }}>
          <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            {filtered.map((s) => <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>)}
          </SelectContent>
        </Select>
        {selected && <p className="text-xs text-muted-foreground mt-1">{selected.description}</p>}
      </div>

      {/* Dynamic Params */}
      {selected?.parameters.map((p) => (
        <div key={p.name}>
          <Label className="text-xs">{p.name}{p.description ? ` — ${p.description}` : ""}</Label>
          {p.type === "bool" ? (
            <div className="flex items-center gap-2 mt-1">
              <Checkbox
                checked={params[p.name] !== undefined ? Boolean(params[p.name]) : Boolean(p.default)}
                onCheckedChange={(v) => handleParamChange(p.name, v === true)}
              />
              <span className="text-sm">{String(params[p.name] ?? p.default)}</span>
            </div>
          ) : (
            <Input
              type="number"
              className="h-8 text-sm"
              value={params[p.name] !== undefined ? Number(params[p.name]) : Number(p.default)}
              min={p.min}
              max={p.max}
              step={p.step || 1}
              onChange={(e) => handleParamChange(p.name, Number(e.target.value))}
            />
          )}
        </div>
      ))}

      {/* Tickers */}
      <div>
        <Label className="text-xs">Ticker Symbol(s)</Label>
        <Input className="h-8 text-sm" value={tickers} onChange={(e) => { setTickers(e.target.value); emitChange({ tickers: e.target.value }); }} placeholder="AAPL, MSFT" />
      </div>

      {/* Capital */}
      <div>
        <Label className="text-xs">Initial Capital ($)</Label>
        <Input type="number" className="h-8 text-sm" value={capital} min={1000} step={1000} onChange={(e) => { setCapital(Number(e.target.value)); emitChange({ initial_capital: Number(e.target.value) }); }} />
      </div>

      {/* Period */}
      <div>
        <Label className="text-xs">Data Period</Label>
        <Select value={period} onValueChange={(v) => { setPeriod(v); emitChange({ period: v }); }}>
          <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            {PERIOD_OPTIONS.map((o) => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {period === "custom" && (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label className="text-xs">Start Date</Label>
            <Input type="date" className="h-8 text-sm" value={startDate} onChange={(e) => { setStartDate(e.target.value); emitChange({ start_date: e.target.value }); }} />
          </div>
          <div>
            <Label className="text-xs">End Date</Label>
            <Input type="date" className="h-8 text-sm" value={endDate} onChange={(e) => { setEndDate(e.target.value); emitChange({ end_date: e.target.value }); }} />
          </div>
        </div>
      )}
    </div>
  );
}
