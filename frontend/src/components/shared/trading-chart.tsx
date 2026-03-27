"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi, ISeriesApi, ColorType } from "lightweight-charts";

interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface LineData {
  time: string;
  value: number;
}

interface VolumeData {
  time: string;
  value: number;
  color?: string;
}

interface TradingChartProps {
  data: CandlestickData[] | LineData[];
  type?: "candlestick" | "line" | "area";
  volumeData?: VolumeData[];
  height?: number;
  className?: string;
  colors?: {
    upColor?: string;
    downColor?: string;
    lineColor?: string;
    areaTopColor?: string;
    areaBottomColor?: string;
  };
}

export function TradingChart({
  data,
  type = "candlestick",
  volumeData,
  height = 400,
  className,
  colors,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "hsl(215, 20%, 55%)",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "hsl(217, 33%, 12%)" },
        horzLines: { color: "hsl(217, 33%, 12%)" },
      },
      crosshair: {
        vertLine: { color: "hsl(142, 71%, 45%)", width: 1, style: 3 },
        horzLine: { color: "hsl(142, 71%, 45%)", width: 1, style: 3 },
      },
      rightPriceScale: {
        borderColor: "hsl(217, 33%, 17%)",
      },
      timeScale: {
        borderColor: "hsl(217, 33%, 17%)",
        timeVisible: true,
      },
      width: containerRef.current.clientWidth,
      height,
    });

    chartRef.current = chart;

    if (type === "candlestick") {
      const series = chart.addCandlestickSeries({
        upColor: colors?.upColor || "#00cc44",
        downColor: colors?.downColor || "#ff3333",
        borderUpColor: colors?.upColor || "#00cc44",
        borderDownColor: colors?.downColor || "#ff3333",
        wickUpColor: colors?.upColor || "#00cc44",
        wickDownColor: colors?.downColor || "#ff3333",
      });
      series.setData(data as CandlestickData[]);
    } else if (type === "line") {
      const series = chart.addLineSeries({
        color: colors?.lineColor || "#00cc44",
        lineWidth: 2,
      });
      series.setData(data as LineData[]);
    } else if (type === "area") {
      const series = chart.addAreaSeries({
        topColor: colors?.areaTopColor || "rgba(0, 204, 68, 0.3)",
        bottomColor: colors?.areaBottomColor || "rgba(0, 204, 68, 0.0)",
        lineColor: colors?.lineColor || "#00cc44",
        lineWidth: 2,
      });
      series.setData(data as LineData[]);
    }

    if (volumeData && volumeData.length > 0) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volumeSeries.setData(
        volumeData.map((v) => ({
          ...v,
          color: v.color || "rgba(0, 204, 68, 0.2)",
        }))
      );
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, type, volumeData, height, colors]);

  return <div ref={containerRef} className={className} />;
}
