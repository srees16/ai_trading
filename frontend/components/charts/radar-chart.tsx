"use client";

import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from "recharts";

interface RadarChartProps {
  data: { layer: string; score: number }[];
  title?: string;
}

export function VerdictRadarChart({ data, title }: RadarChartProps) {
  if (data.length === 0) return null;

  return (
    <div>
      {title && <h5 className="text-sm font-medium text-center mb-1">{title}</h5>}
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="layer" tick={{ fontSize: 11 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Radar name="Score" dataKey="score" stroke="#4299e1" fill="#4299e1" fillOpacity={0.3} />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
