"use client";

import { useState, lazy, Suspense } from "react";
import { indStocksApi, type OptionStrikeData } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber, formatLargeNumber, downloadCsv } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { Loader2, RefreshCw, Link, Activity, BarChart3 } from "lucide-react";
import { toast } from "sonner";

// Lazy-load recharts — only needed when chart data is rendered
const OIChart = lazy(() => import("recharts").then(mod => ({
  default: function OIChartInner({ data }: { data: { strike: number; CE_OI: number; PE_OI: number }[] }) {
    const { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } = mod;
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 12%)" />
          <XAxis dataKey="strike" tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
          <YAxis tick={{ fontSize: 10 }} stroke="hsl(215, 20%, 55%)" />
          <Tooltip contentStyle={{ backgroundColor: "hsl(222, 47%, 9%)", border: "1px solid hsl(217, 33%, 17%)", borderRadius: 8 }} />
          <Legend />
          <Bar dataKey="CE_OI" fill="#22c55e" radius={[2, 2, 0, 0]} />
          <Bar dataKey="PE_OI" fill="#ef4444" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }
})));

const DEFAULT_INDICES: Record<string, number> = {
  NIFTY: 260105, BANKNIFTY: 260361, FINNIFTY: 257801, MIDCPNIFTY: 288009, SENSEX: 265,
};

export default function IndOptionsPage() {
  // Option Chain state
  const [chainIndex, setChainIndex] = useState("NIFTY");
  const [chainLoading, setChainLoading] = useState(false);
  const [spotPrice, setSpotPrice] = useState<number | null>(null);
  const [atmStrike, setAtmStrike] = useState<number | null>(null);
  const [atmIV, setAtmIV] = useState<number | null>(null);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState<string>("");
  const [chain, setChain] = useState<OptionStrikeData[]>([]);
  const [strikeRange, setStrikeRange] = useState(10);
  // Token Lookup
  const [lookupToken, setLookupToken] = useState(9073154);
  const [lookupResult, setLookupResult] = useState<string | null>(null);
  const [lookupLoading, setLookupLoading] = useState(false);

  const loadChain = async (expiry?: string) => {
    setChainLoading(true);
    try {
      const res = await indStocksApi.optionChain(chainIndex, expiry, strikeRange);
      if (res.success) {
        setSpotPrice(res.spot_price);
        setAtmStrike(res.atm_strike);
        setChain(res.chain);
        setExpiries(res.expiries || []);
        if (!expiry && res.expiry) setSelectedExpiry(res.expiry);
        // Compute ATM IV from chain
        const atmRow = res.chain.find((r) => r.strike === res.atm_strike);
        setAtmIV(atmRow ? ((atmRow.ce_iv || 0) + (atmRow.pe_iv || 0)) / 2 : null);
        toast.success(`Loaded ${res.chain.length} strikes for ${chainIndex}`);
      }
    } catch (err: any) {
      toast.error(err?.message || "Failed to load option chain");
    } finally { setChainLoading(false); }
  };

  const handleExpiryChange = (val: string) => {
    setSelectedExpiry(val);
    loadChain(val);
  };

  const chainColumns: ColumnDef<OptionStrikeData>[] = [
    { accessorKey: "ce_iv", header: "CE IV%", cell: ({ row }) => <span className="text-profit">{row.getValue("ce_iv") != null ? formatNumber(row.getValue("ce_iv") as number, 1) + "%" : "—"}</span> },
    { accessorKey: "ce_volume", header: "CE Vol", cell: ({ row }) => row.getValue("ce_volume") != null ? formatLargeNumber(row.getValue("ce_volume") as number) : "—" },
    { accessorKey: "ce_oi", header: "CE OI", cell: ({ row }) => row.getValue("ce_oi") != null ? formatLargeNumber(row.getValue("ce_oi") as number) : "—" },
    { accessorKey: "ce_ltp", header: "CE LTP", cell: ({ row }) => row.getValue("ce_ltp") != null ? `₹${formatNumber(row.getValue("ce_ltp") as number)}` : "—" },
    {
      accessorKey: "strike",
      header: ({ column }) => <SortableHeader column={column} title="Strike" />,
      cell: ({ row }) => {
        const isATM = row.getValue("strike") === atmStrike;
        return <span className={`font-mono font-bold ${isATM ? "text-amber-400 text-lg" : ""}`}>{row.getValue("strike")}{isATM && " ★"}</span>;
      },
    },
    { accessorKey: "pe_ltp", header: "PE LTP", cell: ({ row }) => row.getValue("pe_ltp") != null ? `₹${formatNumber(row.getValue("pe_ltp") as number)}` : "—" },
    { accessorKey: "pe_oi", header: "PE OI", cell: ({ row }) => row.getValue("pe_oi") != null ? formatLargeNumber(row.getValue("pe_oi") as number) : "—" },
    { accessorKey: "pe_volume", header: "PE Vol", cell: ({ row }) => row.getValue("pe_volume") != null ? formatLargeNumber(row.getValue("pe_volume") as number) : "—" },
    { accessorKey: "pe_iv", header: "PE IV%", cell: ({ row }) => <span className="text-loss">{row.getValue("pe_iv") != null ? formatNumber(row.getValue("pe_iv") as number, 1) + "%" : "—"}</span> },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Options & Derivatives</h1>
        <p className="text-sm text-muted-foreground">NSE index option chains, live derivative prices, and instrument lookup</p>
      </div>

      <Tabs defaultValue="chain">
        <TabsList>
          <TabsTrigger value="chain"><BarChart3 className="mr-1 h-3 w-3" />Option Chain</TabsTrigger>
          <TabsTrigger value="quotes"><Activity className="mr-1 h-3 w-3" />Index Quotes</TabsTrigger>
          <TabsTrigger value="lookup"><Link className="mr-1 h-3 w-3" />Token Lookup</TabsTrigger>
        </TabsList>

        {/* Option Chain Tab */}
        <TabsContent value="chain" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-wrap items-end gap-4">
                <div className="w-40">
                  <Label className="text-xs">Index</Label>
                  <Select value={chainIndex} onValueChange={setChainIndex}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.keys(DEFAULT_INDICES).map((idx) => (
                        <SelectItem key={idx} value={idx}>{idx}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-32">
                  <Label className="text-xs">Strike Range (±)</Label>
                  <Input type="number" value={strikeRange} onChange={(e) => setStrikeRange(Number(e.target.value))} min={5} max={30} />
                </div>
                {expiries.length > 0 && (
                  <div className="w-48">
                    <Label className="text-xs">Expiry</Label>
                    <Select value={selectedExpiry} onValueChange={handleExpiryChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {expiries.map((exp) => (
                          <SelectItem key={exp} value={exp}>{exp}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <Button onClick={() => loadChain()} disabled={chainLoading}>
                  {chainLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                  Load Chain
                </Button>
              </div>
            </CardContent>
          </Card>

          {spotPrice !== null && (
            <>
              {(() => {
                // Compute PCR, Max Pain, Total OI from chain data
                const totalCeOi = chain.reduce((s, r) => s + (r.ce_oi || 0), 0);
                const totalPeOi = chain.reduce((s, r) => s + (r.pe_oi || 0), 0);
                const pcr = totalCeOi > 0 ? totalPeOi / totalCeOi : 0;
                // Max pain: strike where total (CE+PE) writer losses are minimized
                let maxPainStrike: number | null = null;
                if (chain.length > 0) {
                  let minPain = Infinity;
                  for (const strike of chain) {
                    let pain = 0;
                    for (const row of chain) {
                      if (row.ce_oi && strike.strike > row.strike)
                        pain += (strike.strike - row.strike) * row.ce_oi;
                      if (row.pe_oi && strike.strike < row.strike)
                        pain += (row.strike - strike.strike) * row.pe_oi;
                    }
                    if (pain < minPain) { minPain = pain; maxPainStrike = strike.strike; }
                  }
                }
                return (
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-6">
                    <MetricCard title="Spot Price" value={`₹${formatNumber(spotPrice)}`} />
                    <MetricCard title="ATM Strike" value={atmStrike ? `₹${formatNumber(atmStrike)}` : "—"} />
                    <MetricCard title="ATM IV" value={atmIV ? `${formatNumber(atmIV, 1)}%` : "—"} />
                    <MetricCard title="PCR (OI)" value={formatNumber(pcr, 2)} changeType={pcr > 1 ? "positive" : pcr < 0.7 ? "negative" : "neutral"} />
                    <MetricCard title="Max Pain" value={maxPainStrike ? `₹${formatNumber(maxPainStrike)}` : "—"} />
                    <MetricCard title="Expiry" value={selectedExpiry || "—"} />
                  </div>
                );
              })()}

              {/* OI Bar Chart */}
              {chain.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-base">Open Interest Distribution</CardTitle></CardHeader>
                  <CardContent>
                    <Suspense fallback={<div className="h-[300px] rounded bg-muted animate-pulse" />}>
                      <OIChart data={chain.map((r) => ({ strike: r.strike, CE_OI: r.ce_oi || 0, PE_OI: r.pe_oi || 0 }))} />
                    </Suspense>
                  </CardContent>
                </Card>
              )}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {chainIndex} Option Chain
                    <Badge variant="outline" className="ml-2">{chain.length} strikes</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <DataTable columns={chainColumns} data={chain} searchKey="strike" pageSize={chain.length > 0 ? chain.length : 20} onExport={() => downloadCsv(
                    ["CE IV%", "CE Vol", "CE OI", "CE LTP", "Strike", "PE LTP", "PE OI", "PE Vol", "PE IV%"],
                    chain.map((r) => [r.ce_iv, r.ce_volume, r.ce_oi, r.ce_ltp, r.strike, r.pe_ltp, r.pe_oi, r.pe_volume, r.pe_iv]),
                    `${chainIndex}_option_chain.csv`
                  )} />
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Index Quotes Tab */}
        <TabsContent value="quotes" className="space-y-4">
          <IndexQuotesPanel />
        </TabsContent>

        {/* Token Lookup Tab */}
        <TabsContent value="lookup" className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-base">Instrument Token Lookup</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-end gap-4">
                <div className="w-48">
                  <Label className="text-xs">Instrument Token</Label>
                  <Input type="number" value={lookupToken} onChange={(e) => setLookupToken(Number(e.target.value))} />
                </div>
                <Button onClick={async () => {
                  setLookupLoading(true);
                  try {
                    const res = await indStocksApi.quotes([String(lookupToken)]);
                    if (res.success && res.quotes.length > 0) {
                      setLookupResult(res.quotes[0].instrument);
                      toast.success(`Found: ${res.quotes[0].instrument}`);
                    } else {
                      setLookupResult("Not found");
                    }
                  } catch { setLookupResult("Error looking up token"); }
                  finally { setLookupLoading(false); }
                }} disabled={lookupLoading}>
                  {lookupLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Link className="mr-2 h-4 w-4" />}
                  Lookup
                </Button>
              </div>
              {lookupResult && (
                <div className="rounded border p-3">
                  <p className="text-sm text-muted-foreground">Result:</p>
                  <p className="font-mono text-lg font-bold">{lookupResult}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Separate component for Index Quotes to manage its own state
function IndexQuotesPanel() {
  const [selectedIndices, setSelectedIndices] = useState(["NIFTY", "BANKNIFTY", "FINNIFTY"]);
  const [quotes, setQuotes] = useState<Record<string, { ltp: number; change: number; changePct: number }>>({});
  const [loading, setLoading] = useState(false);

  const fetchQuotes = async () => {
    setLoading(true);
    try {
      const instruments = selectedIndices.map((idx) => `NSE:${idx}`);
      const res = await indStocksApi.quotes(instruments);
      if (res.success) {
        const q: typeof quotes = {};
        res.quotes.forEach((qt) => {
          q[qt.instrument] = { ltp: qt.last_price || 0, change: qt.change || 0, changePct: qt.change_pct || 0 };
        });
        setQuotes(q);
      }
    } catch (err: any) { toast.error(err?.message || "Failed to fetch quotes"); }
    finally { setLoading(false); }
  };

  return (
    <>
      <Card>
        <CardContent className="flex items-end gap-4 pt-6">
          <div className="flex-1">
            <Label className="text-xs">Select Indices</Label>
            <div className="flex flex-wrap gap-2 mt-1">
              {Object.keys(DEFAULT_INDICES).map((idx) => (
                <Badge
                  key={idx}
                  variant={selectedIndices.includes(idx) ? "default" : "outline"}
                  className="cursor-pointer select-none"
                  onClick={() => setSelectedIndices((prev) => prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx])}
                >
                  {idx}
                </Badge>
              ))}
            </div>
          </div>
          <Button onClick={fetchQuotes} disabled={loading || selectedIndices.length === 0}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
            Refresh
          </Button>
        </CardContent>
      </Card>
      {Object.keys(quotes).length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
          {Object.entries(quotes).map(([name, data]) => (
            <Card key={name}>
              <CardContent className="pt-6">
                <p className="text-xs text-muted-foreground">{name}</p>
                <p className="text-2xl font-bold">₹{formatNumber(data.ltp)}</p>
                <p className={`text-sm font-medium ${data.changePct >= 0 ? "text-profit" : "text-loss"}`}>
                  {data.changePct >= 0 ? "+" : ""}{formatNumber(data.changePct, 2)}% ({data.change >= 0 ? "+" : ""}{formatNumber(data.change)})
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
