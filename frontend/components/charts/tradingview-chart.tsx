"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, type IChartApi } from "lightweight-charts";

interface TradingViewChartProps {
  data: { time: string; open: number; high: number; low: number; close: number; volume?: number }[];
  type?: "candlestick" | "line" | "area";
  height?: number;
  title?: string;
}

export function TradingViewChart({ data, type = "candlestick", height = 400, title }: TradingViewChartProps) {
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
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      rightPriceScale: { borderColor: "#ddd" },
      timeScale: { borderColor: "#ddd" },
    });

    if (type === "candlestick") {
      const series = chart.addCandlestickSeries({
        upColor: "#00cc44",
        downColor: "#ff3333",
        borderDownColor: "#ff3333",
        borderUpColor: "#00cc44",
        wickDownColor: "#ff3333",
        wickUpColor: "#00cc44",
      });
      series.setData(data as any);
    } else if (type === "line") {
      const series = chart.addLineSeries({ color: "#4299e1", lineWidth: 2 });
      series.setData(data.map((d) => ({ time: d.time, value: d.close })) as any);
    } else {
      const series = chart.addAreaSeries({
        topColor: "rgba(66, 153, 225, 0.4)",
        bottomColor: "rgba(66, 153, 225, 0.05)",
        lineColor: "#4299e1",
        lineWidth: 2,
      });
      series.setData(data.map((d) => ({ time: d.time, value: d.close })) as any);
    }

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
  }, [data, type, height]);

  return (
    <div>
      {title && <h4 className="text-sm font-semibold mb-2">{title}</h4>}
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
