"use client";

import { cn } from "@/lib/utils";
import { VIX_CAUTION_THRESHOLD, VIX_PANIC_THRESHOLD } from "@/lib/constants";
import { useMacroSnapshot, useTickerPrices } from "@/hooks/use-macro";

interface RibbonVixBarProps {
  tickers?: { symbol: string; price: number; change_pct: number }[];
  symbols?: string[];
  vix?: number;
  market?: "US" | "IND";
}

export function RibbonVixBar({ tickers = [], symbols = [], vix, market }: RibbonVixBarProps) {
  const inferredMarket = market ?? (symbols.some((s) => ["NIFTY50", "BANKNIFTY", "SENSEX"].includes(s)) ? "IND" : "US");
  const macroQ = useMacroSnapshot(inferredMarket);
  const macro = macroQ.data;

  // Fetch live prices for symbols (refreshes every 5s during market hours, 60s after)
  const pricesQ = useTickerPrices(
    tickers.length > 0 ? [] : symbols,
    inferredMarket,
  );

  const effectiveVix = vix ?? macro?.vix ?? undefined;

  const isLoadingPrices = tickers.length === 0 && symbols.length > 0 && pricesQ.prices.length === 0 && pricesQ.isLoading;

  const displayTickers = tickers.length > 0
    ? tickers
    : pricesQ.prices.length > 0 ? pricesQ.prices : [];

  const vixColor = effectiveVix === undefined ? "bg-gray-200" :
    effectiveVix > VIX_PANIC_THRESHOLD ? "bg-trade-red" :
    effectiveVix > VIX_CAUTION_THRESHOLD ? "bg-trade-yellow" : "bg-trade-green";

  const vixLabel = effectiveVix === undefined ? "N/A" :
    effectiveVix > VIX_PANIC_THRESHOLD ? "Panic" :
    effectiveVix > VIX_CAUTION_THRESHOLD ? "Caution" : "Calm";

  const sentimentColor =
    macro?.macro_sentiment_label === "greedy" ? "bg-green-600" :
    macro?.macro_sentiment_label === "fearful" ? "bg-red-600" : "bg-yellow-600";

  return (
    <div className="space-y-1 mb-2">
      {/* Row 1: Ticker Ribbon */}
      {isLoadingPrices ? (
        <div className="ticker-ribbon rounded bg-secondary/50 py-1 px-2">
          <div className="ticker-ribbon-inner gap-6 text-xs">
            {Array.from({ length: 12 }).map((_, i) => (
              <span key={i} className="inline-flex items-center gap-1 mr-6">
                <span className="h-3 w-12 bg-muted-foreground/20 rounded animate-pulse" />
                <span className="h-3 w-14 bg-muted-foreground/20 rounded animate-pulse" />
                <span className="h-3 w-10 bg-muted-foreground/20 rounded animate-pulse" />
              </span>
            ))}
          </div>
        </div>
      ) : displayTickers.length > 0 && (
        <div className="ticker-ribbon rounded bg-secondary/50 py-1 px-2">
          <div className="ticker-ribbon-inner gap-6 text-xs">
            {[...displayTickers, ...displayTickers].map((t, i) => (
              <span key={i} className="inline-flex items-center gap-1 mr-6">
                <span className="font-medium">{t.symbol}</span>
                {t.price > 0 && <span>{inferredMarket === "IND" ? "₹" : "$"}{t.price.toFixed(2)}</span>}
                {t.change_pct !== 0 && (
                  <span className={t.change_pct >= 0 ? "pnl-positive" : "pnl-negative"}>
                    {t.change_pct >= 0 ? "+" : ""}{t.change_pct.toFixed(2)}%
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Row 2: VIX + Macro Indicators */}
      <div className="flex items-center gap-2 flex-wrap text-xs">
        {/* VIX Pill */}
        {effectiveVix !== undefined && (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border bg-background">
            <span className="font-medium">{macro?.vix_label ?? "VIX"}</span>
            <span className="font-bold">{effectiveVix.toFixed(1)}</span>
            <span className={cn("px-1.5 py-0.5 rounded-full text-white text-[0.6rem] font-bold", vixColor)}>
              {vixLabel}
            </span>
          </div>
        )}

        {/* Sentiment Pill */}
        {macro?.macro_sentiment_label && (
          <span className={cn("px-2 py-0.5 rounded-full text-white text-[0.65rem] font-bold uppercase", sentimentColor)}>
            {macro.macro_sentiment_label}
          </span>
        )}

        {/* Index Price */}
        {macro?.index_price != null && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded border bg-background">
            <span className="text-muted-foreground">{macro.index_name}</span>
            <span className="font-bold">{macro.index_price.toLocaleString(undefined, { maximumFractionDigits: 1 })}</span>
            {macro.index_change_pct != null && (
              <span className={macro.index_change_pct >= 0 ? "pnl-positive" : "pnl-negative"}>
                {macro.index_change_pct >= 0 ? "+" : ""}{macro.index_change_pct.toFixed(2)}%
              </span>
            )}
          </div>
        )}

        {/* 10Y Yield */}
        {macro?.us_10y_yield != null && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded border bg-background">
            <span className="text-muted-foreground">10Y</span>
            <span className="font-semibold">{macro.us_10y_yield.toFixed(2)}%</span>
          </div>
        )}

        {/* Gold */}
        {macro?.gold_price != null && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded border bg-background">
            <span className="text-muted-foreground">Gold</span>
            <span className="font-semibold">${macro.gold_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          </div>
        )}

        {/* Crude Oil */}
        {macro?.crude_oil_price != null && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded border bg-background">
            <span className="text-muted-foreground">Crude</span>
            <span className="font-semibold">${macro.crude_oil_price.toFixed(1)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
