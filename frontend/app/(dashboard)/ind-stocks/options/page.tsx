"use client";

import { useState } from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { useIndexQuotes, useOptionChain, useOptionExpiries } from "@/hooks/use-options";
import { INDEX_OPTIONS, NIFTY_50_TICKERS } from "@/lib/constants";
import { formatCurrency } from "@/lib/utils";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";

type MoneynessFilter = "ALL" | "ITM" | "ATM" | "OTM";

export default function INDOptionsPage() {
  const [symbol, setSymbol] = useState("NIFTY");
  const [expiry, setExpiry] = useState("");
  const [moneyness, setMoneyness] = useState<MoneynessFilter>("ALL");
  const indexQ = useIndexQuotes();
  const expiriesQ = useOptionExpiries(symbol);
  const chainQ = useOptionChain(symbol, expiry);

  const indices = indexQ.data ?? [];
  const chain = chainQ.data ?? [];
  const filteredChain = moneyness === "ALL" ? chain : chain.filter((r) => r.moneyness === moneyness);

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NIFTY_50_TICKERS} market="IND" />
      {/* Index quotes ribbon */}
      <MetricsGrid>
        {indices.map((idx) => (
          <MetricCard
            key={idx.index}
            label={idx.index}
            value={formatCurrency(idx.ltp)}
            color={idx.change >= 0 ? "text-green-500" : "text-red-500"}
            deltaLabel={`${idx.change >= 0 ? "+" : ""}${idx.change.toFixed(2)} (${idx.change_pct.toFixed(2)}%)`}
          />
        ))}
      </MetricsGrid>

      {/* Controls */}
      <div className="flex gap-4 items-end flex-wrap">
        <div>
          <Label className="text-xs">Symbol</Label>
          <Select value={symbol} onValueChange={(v) => { setSymbol(v); setExpiry(""); }}>
            <SelectTrigger className="h-8 text-sm w-40"><SelectValue /></SelectTrigger>
            <SelectContent>
              {INDEX_OPTIONS.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Expiry</Label>
          <Select value={expiry} onValueChange={setExpiry} disabled={!expiriesQ.data?.length}>
            <SelectTrigger className="h-8 text-sm w-40"><SelectValue placeholder="Select expiry" /></SelectTrigger>
            <SelectContent>
              {expiriesQ.data?.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Moneyness</Label>
          <Select value={moneyness} onValueChange={(v) => setMoneyness(v as MoneynessFilter)}>
            <SelectTrigger className="h-8 text-sm w-28"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="ALL">All</SelectItem>
              <SelectItem value="ITM">ITM</SelectItem>
              <SelectItem value="ATM">ATM</SelectItem>
              <SelectItem value="OTM">OTM</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Option Chain */}
      {chainQ.isLoading && <Spinner />}
      {filteredChain.length > 0 && (
        <div className="content-panel p-4 overflow-x-auto">
          <h4 className="text-sm font-semibold mb-2">Option Chain — {symbol} {expiry}</h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th colSpan={9} className="py-2 text-center border-r bg-green-500/5">CALLS</th>
                <th className="py-2 px-3">Strike</th>
                <th colSpan={9} className="py-2 text-center border-l bg-red-500/5">PUTS</th>
              </tr>
              <tr className="border-b text-muted-foreground text-xs">
                <th className="py-1 px-1">OI</th><th className="py-1 px-1">ΔOI</th><th className="py-1 px-1">Vol</th><th className="py-1 px-1">IV</th><th className="py-1 px-1">LTP</th>
                <th className="py-1 px-1">Δ</th><th className="py-1 px-1">Γ</th><th className="py-1 px-1">Θ</th><th className="py-1 px-1 border-r">V</th>
                <th className="py-1 px-3 font-bold">Strike</th>
                <th className="py-1 px-1 border-l">LTP</th><th className="py-1 px-1">IV</th><th className="py-1 px-1">Vol</th><th className="py-1 px-1">ΔOI</th><th className="py-1 px-1">OI</th>
                <th className="py-1 px-1">Δ</th><th className="py-1 px-1">Γ</th><th className="py-1 px-1">Θ</th><th className="py-1 px-1">V</th>
              </tr>
            </thead>
            <tbody>
              {filteredChain.map((row) => (
                <tr key={row.strike} className={`border-b hover:bg-muted/30 ${row.moneyness === "ATM" ? "bg-yellow-500/5" : ""}`}>
                  <td className="py-1 px-1 text-right">{row.ce_oi.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right">{row.ce_chg_oi.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right">{row.ce_volume.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right">{row.ce_iv.toFixed(1)}</td>
                  <td className="py-1 px-1 text-right font-mono">{row.ce_ltp.toFixed(2)}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.ce_delta?.toFixed(2) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.ce_gamma?.toFixed(4) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.ce_theta?.toFixed(2) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground border-r">{row.ce_vega?.toFixed(2) ?? "-"}</td>
                  <td className="py-1 px-3 text-center font-bold bg-muted/30">{row.strike}</td>
                  <td className="py-1 px-1 text-right border-l font-mono">{row.pe_ltp.toFixed(2)}</td>
                  <td className="py-1 px-1 text-right">{row.pe_iv.toFixed(1)}</td>
                  <td className="py-1 px-1 text-right">{row.pe_volume.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right">{row.pe_chg_oi.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right">{row.pe_oi.toLocaleString()}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.pe_delta?.toFixed(2) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.pe_gamma?.toFixed(4) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.pe_theta?.toFixed(2) ?? "-"}</td>
                  <td className="py-1 px-1 text-right text-muted-foreground">{row.pe_vega?.toFixed(2) ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
