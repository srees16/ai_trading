"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from "recharts";
import { HEALTH_COLORS } from "@/lib/constants";
import { getZScoreStatus, getMScoreStatus, getFScoreStatus } from "@/lib/utils";
import type { FundamentalRow } from "@/components/tables/fundamental-table";

interface FundamentalChartsProps {
  data: FundamentalRow[];
}

export function FundamentalCharts({ data: rawData }: FundamentalChartsProps) {
  const data = rawData.map((d) => ({ ticker: d.ticker, z: d.z_score, m: d.m_score, f: d.f_score }));
  if (data.length === 0) return <p className="text-sm text-muted-foreground">No fundamental data</p>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Z-Score */}
      <div>
        <h4 className="text-sm font-semibold mb-2">Altman Z-Score</h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ticker" tick={{ fontSize: 10 }} />
            <YAxis />
            <ReferenceLine y={2.99} stroke="#00cc44" strokeDasharray="5 5" label="Safe" />
            <ReferenceLine y={1.81} stroke="#ffcc00" strokeDasharray="5 5" label="Grey" />
            <Tooltip />
            <Bar dataKey="z" name="Z-Score">
              {data.map((d, i) => (
                <Cell key={i} fill={HEALTH_COLORS[getZScoreStatus(d.z)] || "#999"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* M-Score */}
      <div>
        <h4 className="text-sm font-semibold mb-2">Beneish M-Score</h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ticker" tick={{ fontSize: 10 }} />
            <YAxis />
            <ReferenceLine y={-2.22} stroke="#ff3333" strokeDasharray="5 5" label="-2.22" />
            <Tooltip />
            <Bar dataKey="m" name="M-Score">
              {data.map((d, i) => (
                <Cell key={i} fill={HEALTH_COLORS[getMScoreStatus(d.m)] || "#999"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* F-Score */}
      <div>
        <h4 className="text-sm font-semibold mb-2">Piotroski F-Score</h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ticker" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 10]} />
            <ReferenceLine y={8} stroke="#00cc44" strokeDasharray="5 5" label="Strong" />
            <ReferenceLine y={5} stroke="#ffcc00" strokeDasharray="5 5" label="Moderate" />
            <Tooltip />
            <Bar dataKey="f" name="F-Score">
              {data.map((d, i) => (
                <Cell key={i} fill={HEALTH_COLORS[getFScoreStatus(d.f)] || "#999"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
