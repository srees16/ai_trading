"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { SENTIMENT_COLORS } from "@/lib/constants";
import type { TradingSignal } from "@/lib/types";

interface SentimentBarChartProps {
  signals: TradingSignal[];
}

export function SentimentBarChart({ signals }: SentimentBarChartProps) {
  const data = signals.map((s) => ({
    ticker: s.news_item.ticker,
    confidence: s.news_item.sentiment_confidence,
    sentiment: s.news_item.sentiment_label,
  }));

  if (data.length === 0) return <p className="text-sm text-muted-foreground">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 10, right: 30, bottom: 30, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="ticker" tick={{ fontSize: 10 }} />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Bar dataKey="confidence" name="Sentiment Confidence">
          {data.map((entry, i) => (
            <Cell key={i} fill={SENTIMENT_COLORS[entry.sentiment] || "#999"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
