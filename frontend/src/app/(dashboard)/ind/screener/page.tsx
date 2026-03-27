"use client";

import { useState } from "react";
import { pipelineApi, type PipelineResponse, type VerdictItem, type PlanItem, type OrderItem } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { Separator } from "@/components/ui/separator";
import { formatNumber, getDecisionBadgeClass } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { Loader2, Play, ScanSearch, Zap, ArrowUpDown, Download, ShieldCheck, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

export default function IndScreenerPage() {
  // Screener config
  const [minPrice, setMinPrice] = useState(100);
  const [minVolume, setMinVolume] = useState(500000);
  const [minBeta, setMinBeta] = useState(1.0);
  const [workers, setWorkers] = useState(8);
  const [indexMode, setIndexMode] = useState(true);
  // Risk config
  const [capital, setCapital] = useState(500000);
  const [riskPct, setRiskPct] = useState(2);
  const [maxTrades, setMaxTrades] = useState(10);
  const [minRR, setMinRR] = useState(2.0);
  const [slMethod, setSlMethod] = useState("tighter");
  // Breakout config
  const [pbTolerance, setPbTolerance] = useState(2);
  const [bvMult, setBvMult] = useState(1.5);
  const [bvLookback, setBvLookback] = useState(20);
  // State
  const [loading, setLoading] = useState(false);
  const [fullLoading, setFullLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [autoPlace, setAutoPlace] = useState(false);

  const handleScreen = async () => {
    setLoading(true);
    setProgress(20);
    try {
      setProgress(50);
      const res = await pipelineApi.screen({ index_mode: indexMode, auto_place: false });
      setResult(res);
      setProgress(100);
      toast.success(`Screened ${res.screened_count} stocks from ${res.universe_size} universe`);
    } catch (err: any) {
      toast.error(err?.message || "Screening failed");
    } finally { setLoading(false); setProgress(0); }
  };

  const handleFullPipeline = async () => {
    setFullLoading(true);
    setProgress(10);
    try {
      setProgress(30);
      const res = await pipelineApi.full({ index_mode: indexMode, auto_place: autoPlace });
      setResult(res);
      setProgress(100);
      toast.success(`Pipeline complete — ${res.orders_placed} orders placed`);
    } catch (err: any) {
      toast.error(err?.message || "Pipeline failed");
    } finally { setFullLoading(false); setProgress(0); }
  };

  const verdictColumns: ColumnDef<VerdictItem>[] = [
    { accessorKey: "ticker", header: ({ column }) => <SortableHeader column={column} title="Ticker" />, cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("ticker")}</span> },
    { accessorKey: "score", header: ({ column }) => <SortableHeader column={column} title="Score" />, cell: ({ row }) => <span className="font-bold">{formatNumber(row.getValue("score") as number, 3)}</span> },
    { accessorKey: "classification", header: ({ column }) => <SortableHeader column={column} title="Verdict" />, cell: ({ row }) => <Badge className={getDecisionBadgeClass(row.getValue("classification") as string)}>{row.getValue("classification")}</Badge> },
    { accessorKey: "confidence", header: ({ column }) => <SortableHeader column={column} title="Confidence" />, cell: ({ row }) => `${formatNumber((row.getValue("confidence") as number) * 100, 1)}%` },
  ];

  const planColumns: ColumnDef<PlanItem>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => <Badge variant={row.getValue("side") === "BUY" ? "default" : "destructive"}>{row.getValue("side")}</Badge> },
    { accessorKey: "entry_price", header: "Entry", cell: ({ row }) => `₹${formatNumber(row.getValue("entry_price") as number)}` },
    { accessorKey: "stop_loss", header: "SL", cell: ({ row }) => `₹${formatNumber(row.getValue("stop_loss") as number)}` },
    { accessorKey: "target_price", header: "Target", cell: ({ row }) => `₹${formatNumber(row.getValue("target_price") as number)}` },
    { accessorKey: "quantity", header: "Qty" },
    { accessorKey: "rr_ratio", header: "R:R", cell: ({ row }) => formatNumber(row.getValue("rr_ratio") as number, 1) },
  ];

  const orderColumns: ColumnDef<OrderItem>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => <Badge variant={row.getValue("side") === "BUY" ? "default" : "destructive"}>{row.getValue("side")}</Badge> },
    { accessorKey: "quantity", header: "Qty" },
    { accessorKey: "order_id", header: "Order ID", cell: ({ row }) => <span className="font-mono text-xs">{row.getValue("order_id") || "—"}</span> },
    { accessorKey: "success", header: "Status", cell: ({ row }) => row.getValue("success") ? <Badge className="bg-profit/20 text-profit">Placed</Badge> : <Badge variant="destructive">Failed</Badge> },
    { accessorKey: "error", header: "Error", cell: ({ row }) => <span className="text-xs text-loss">{(row.getValue("error") as string) || ""}</span> },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">NSE Stock Screener</h1>
        <p className="text-sm text-muted-foreground">Screen NSE stocks with breakout detection, risk management, and auto-order placement</p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Screener Config */}
        <Card>
          <CardHeader><CardTitle className="text-base">Screener Config</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div><Label className="text-xs">Min Price (₹)</Label><Input type="number" value={minPrice} onChange={(e) => setMinPrice(Number(e.target.value))} step={10} /></div>
              <div><Label className="text-xs">Min Avg Volume</Label><Input type="number" value={minVolume} onChange={(e) => setMinVolume(Number(e.target.value))} step={50000} /></div>
              <div><Label className="text-xs">Min Beta</Label><Input type="number" value={minBeta} onChange={(e) => setMinBeta(Number(e.target.value))} step={0.1} /></div>
              <div><Label className="text-xs">Workers</Label><Input type="number" value={workers} onChange={(e) => setWorkers(Number(e.target.value))} min={1} max={16} /></div>
            </div>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={indexMode} onChange={(e) => setIndexMode(e.target.checked)} className="rounded" />Index mode</label>
          </CardContent>
        </Card>

        {/* Risk Config */}
        <Card>
          <CardHeader><CardTitle className="text-base">Risk Config</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div><Label className="text-xs">Capital (₹)</Label><Input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} step={50000} /></div>
              <div><Label className="text-xs">Risk per trade %</Label><Input type="number" value={riskPct} onChange={(e) => setRiskPct(Number(e.target.value))} min={1} max={5} /></div>
              <div><Label className="text-xs">Max open trades</Label><Input type="number" value={maxTrades} onChange={(e) => setMaxTrades(Number(e.target.value))} min={1} max={50} /></div>
              <div><Label className="text-xs">Min R:R ratio</Label><Input type="number" value={minRR} onChange={(e) => setMinRR(Number(e.target.value))} step={0.5} /></div>
            </div>
            <div><Label className="text-xs">Stop-Loss Method</Label>
              <Select value={slMethod} onValueChange={setSlMethod}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="tighter">Tighter</SelectItem>
                  <SelectItem value="ma50">MA50</SelectItem>
                  <SelectItem value="swing_low">Swing Low</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Breakout Config */}
        <Card>
          <CardHeader><CardTitle className="text-base">Breakout Config</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div><Label className="text-xs">Pullback tolerance %</Label><Input type="number" value={pbTolerance} onChange={(e) => setPbTolerance(Number(e.target.value))} min={1} max={5} /></div>
            <div><Label className="text-xs">Volume multiplier</Label><Input type="number" value={bvMult} onChange={(e) => setBvMult(Number(e.target.value))} step={0.1} /></div>
            <div><Label className="text-xs">Lookback days</Label><Input type="number" value={bvLookback} onChange={(e) => setBvLookback(Number(e.target.value))} step={5} /></div>
            <Separator className="my-2" />
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={autoPlace} onChange={(e) => setAutoPlace(e.target.checked)} className="rounded" /><span className="text-amber-400">Enable live orders</span></label>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <Button onClick={handleScreen} disabled={loading || fullLoading} size="lg" className="flex-1">
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ScanSearch className="mr-2 h-4 w-4" />}
          {loading ? "Screening..." : "Screen Stocks"}
        </Button>
        <Button onClick={handleFullPipeline} disabled={loading || fullLoading} size="lg" variant="outline" className="flex-1">
          {fullLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
          {fullLoading ? "Running Pipeline..." : "Full Pipeline"}
        </Button>
      </div>

      {(loading || fullLoading) && <Progress value={progress} />}

      {/* Results */}
      {result && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCard title="Universe" value={String(result.universe_size)} />
            <MetricCard title="Passed Screen" value={String(result.screened_count)} icon={<ShieldCheck className="h-4 w-4" />} />
            <MetricCard title="Buy Signals" value={String(result.buy_signals)} changeType="positive" />
            <MetricCard title="Orders Placed" value={String(result.orders_placed)} icon={result.orders_failed > 0 ? <AlertTriangle className="h-4 w-4 text-loss" /> : undefined} />
          </div>

          <Tabs defaultValue="verdicts">
            <TabsList>
              <TabsTrigger value="verdicts">Verdicts ({result.verdicts?.length || 0})</TabsTrigger>
              <TabsTrigger value="plans">Trade Plans ({result.plans?.length || 0})</TabsTrigger>
              <TabsTrigger value="orders">Orders ({result.orders?.length || 0})</TabsTrigger>
            </TabsList>
            <TabsContent value="verdicts">
              <Card><CardContent className="pt-6">
                {result.verdicts?.length ? <DataTable columns={verdictColumns} data={result.verdicts} searchKey="ticker" /> : <p className="text-sm text-muted-foreground">No verdicts generated</p>}
              </CardContent></Card>
            </TabsContent>
            <TabsContent value="plans">
              <Card><CardContent className="pt-6">
                {result.plans?.length ? <DataTable columns={planColumns} data={result.plans} searchKey="symbol" /> : <p className="text-sm text-muted-foreground">No trade plans generated</p>}
              </CardContent></Card>
            </TabsContent>
            <TabsContent value="orders">
              <Card><CardContent className="pt-6">
                {result.orders?.length ? <DataTable columns={orderColumns} data={result.orders} searchKey="symbol" /> : <p className="text-sm text-muted-foreground">No orders placed</p>}
              </CardContent></Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
