"use client";

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from "recharts";
import { DECISION_COLORS } from "@/lib/constants";
import type { TradingSignal } from "@/lib/types";

interface ScoreScatterChartProps {
  signals: TradingSignal[];
}

export function ScoreScatterChart({ signals }: ScoreScatterChartProps) {
  const data = signals.map((s, i) => ({
    ticker: s.news_item.ticker,
    score: s.decision_score,
    decision: s.decision,
    index: i,
  }));

  if (data.length === 0) return <p className="text-sm text-muted-foreground">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="ticker" name="Ticker" type="category" allowDuplicatedCategory={false} tick={{ fontSize: 10 }} />
        <YAxis dataKey="score" name="Score" />
        <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
        <Scatter data={data} name="Signals">
          {data.map((entry, i) => (
            <Cell key={i} fill={DECISION_COLORS[entry.decision] || "#999"} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
