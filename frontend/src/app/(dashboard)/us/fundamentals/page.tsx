"use client";

import { useState } from "react";
import { usStocksApi, type StockMetrics } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";

function getZScoreHealth(z: number | null): { label: string; color: string } {
  if (z == null) return { label: "N/A", color: "#666" };
  if (z > 2.99) return { label: "Safe", color: "#00cc44" };
  if (z > 1.81) return { label: "Grey Zone", color: "#ffcc00" };
  return { label: "Distress", color: "#ff3333" };
}

function getMScoreHealth(m: number | null): { label: string; color: string } {
  if (m == null) return { label: "N/A", color: "#666" };
  if (m < -2.22) return { label: "Unlikely Manipulator", color: "#00cc44" };
  return { label: "Possible Manipulator", color: "#ff3333" };
}

function getFScoreHealth(f: number | null): { label: string; color: string } {
  if (f == null) return { label: "N/A", color: "#666" };
  if (f >= 7) return { label: "Strong", color: "#00cc44" };
  if (f >= 4) return { label: "Moderate", color: "#ffcc00" };
  return { label: "Weak", color: "#ff3333" };
}

export default function USFundamentalsPage() {
  // This page reads from session/signals data or fetches metrics directly
  const [metrics, setMetrics] = useState<StockMetrics[]>([]);
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL, TSLA, NVDA");
  const [loading, setLoading] = useState(false);

  const handleFetch = async () => {
    const tickerList = tickers.split(/[,\n]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickerList.length === 0) return;
    setLoading(true);
    try {
      const res = await usStocksApi.metrics(tickerList);
      if (res.success) setMetrics(res.metrics);
    } catch {} finally {
      setLoading(false);
    }
  };

  const zData = metrics.map((m) => ({
    ticker: m.ticker,
    value: m.altman_z_score ?? 0,
    ...getZScoreHealth(m.altman_z_score),
  }));

  const mData = metrics.map((m) => ({
    ticker: m.ticker,
    value: m.beneish_m_score ?? 0,
    ...getMScoreHealth(m.beneish_m_score),
  }));

  const fData = metrics.map((m) => ({
    ticker: m.ticker,
    value: m.piotroski_f_score ?? 0,
    ...getFScoreHealth(m.piotroski_f_score),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Fundamentals</h1>
          <p className="text-sm text-muted-foreground">Altman Z-Score, Beneish M-Score, Piotroski F-Score</p>
        </div>
        <div className="flex items-center gap-2">
          <input
            className="rounded border border-input bg-background px-3 py-1.5 text-sm"
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            placeholder="AAPL, MSFT..."
          />
          <button
            onClick={handleFetch}
            disabled={loading}
            className="rounded bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {loading ? "Loading..." : "Fetch"}
          </button>
        </div>
      </div>

      {/* Score Interpretation */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Score Interpretation Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-xs">
            <div>
              <p className="mb-1 font-semibold">Altman Z-Score</p>
              <p><Badge className="badge-strong-buy">{">"} 2.99</Badge> Safe Zone</p>
              <p><Badge className="badge-hold">1.81 – 2.99</Badge> Grey Zone</p>
              <p><Badge className="badge-strong-sell">{"<"} 1.81</Badge> Distress Zone</p>
            </div>
            <div>
              <p className="mb-1 font-semibold">Beneish M-Score</p>
              <p><Badge className="badge-strong-buy">{"<"} -2.22</Badge> Unlikely Manipulator</p>
              <p><Badge className="badge-strong-sell">{">"} -2.22</Badge> Possible Manipulator</p>
            </div>
            <div>
              <p className="mb-1 font-semibold">Piotroski F-Score</p>
              <p><Badge className="badge-strong-buy">7 – 9</Badge> Strong</p>
              <p><Badge className="badge-hold">4 – 6</Badge> Moderate</p>
              <p><Badge className="badge-strong-sell">0 – 3</Badge> Weak</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {metrics.length > 0 && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Altman Z-Score */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Altman Z-Score</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={zData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
                  <XAxis dataKey="ticker" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
                  <YAxis stroke="hsl(215, 20%, 55%)" />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
                  <ReferenceLine y={2.99} stroke="#00cc44" strokeDasharray="3 3" label="Safe" />
                  <ReferenceLine y={1.81} stroke="#ff3333" strokeDasharray="3 3" label="Distress" />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {zData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Beneish M-Score */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Beneish M-Score</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
                  <XAxis dataKey="ticker" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
                  <YAxis stroke="hsl(215, 20%, 55%)" />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
                  <ReferenceLine y={-2.22} stroke="#ff3333" strokeDasharray="3 3" label="-2.22" />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {mData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Piotroski F-Score */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Piotroski F-Score</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={fData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
                  <XAxis dataKey="ticker" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
                  <YAxis stroke="hsl(215, 20%, 55%)" domain={[0, 9]} />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {fData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
