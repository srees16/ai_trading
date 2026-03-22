"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { DECISION_COLORS } from "@/lib/constants";
import { decisionSummary } from "@/lib/utils";
import type { TradingSignal } from "@/lib/types";

interface DecisionPieChartProps {
  signals: TradingSignal[];
}

export function DecisionPieChart({ signals }: DecisionPieChartProps) {
  const counts = decisionSummary(signals);
  const data = Object.entries(counts)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: k.replace("_", " "), value: v, key: k }));

  if (data.length === 0) return <p className="text-sm text-muted-foreground">No signals</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          outerRadius={100}
          innerRadius={40}
          paddingAngle={2}
          dataKey="value"
          label={({ name, value }) => `${name}: ${value}`}
        >
          {data.map((entry) => (
            <Cell key={entry.key} fill={DECISION_COLORS[entry.key] || "#999"} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
