"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, type IChartApi } from "lightweight-charts";
import type { EquityPoint } from "@/lib/types";

interface EquityCurveChartProps {
  data: EquityPoint[];
  height?: number;
}

export function EquityCurveChart({ data, height = 350 }: EquityCurveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#333",
      },
      grid: {
        vertLines: { color: "#e8e8e8" },
        horzLines: { color: "#e8e8e8" },
      },
      rightPriceScale: { borderColor: "#ccc" },
      timeScale: { borderColor: "#ccc" },
    });

    const series = chart.addAreaSeries({
      topColor: "rgba(66, 153, 225, 0.4)",
      bottomColor: "rgba(66, 153, 225, 0.05)",
      lineColor: "#4299e1",
      lineWidth: 2,
    });

    series.setData(
      data.map((d) => ({
        time: d.date as string,
        value: d.value,
      }))
    );

    chart.timeScale().fitContent();
    chartRef.current = chart;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) chart.applyOptions({ width });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height]);

  if (data.length === 0) return <p className="text-sm text-muted-foreground">No equity data</p>;

  return <div ref={containerRef} className="w-full" />;
}
