"use client";

import { useState, useEffect } from "react";
import { indStocksApi, type OrderInfo, type PositionData, type HoldingData, type PlaceOrderRequest } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { formatNumber } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { Loader2, RefreshCw, ShoppingCart, X, Package, Briefcase, ListOrdered, CheckCircle, XCircle } from "lucide-react";
import { toast } from "sonner";

export default function IndTradingPage() {
  const [kiteStatus, setKiteStatus] = useState<{ authenticated: boolean; user_id: string | null; market_open: boolean } | null>(null);
  const [orders, setOrders] = useState<OrderInfo[]>([]);
  const [positions, setPositions] = useState<{ net: PositionData[]; day: PositionData[] }>({ net: [], day: [] });
  const [holdings, setHoldings] = useState<{ holdings: HoldingData[]; total_pnl: number; day_pnl: number; total_investment: number; total_current_value: number }>({ holdings: [], total_pnl: 0, day_pnl: 0, total_investment: 0, total_current_value: 0 });
  const [loading, setLoading] = useState(false);

  // Order form
  const [orderSymbol, setOrderSymbol] = useState("");
  const [orderType, setOrderType] = useState("MARKET");
  const [txnType, setTxnType] = useState("BUY");
  const [orderQty, setOrderQty] = useState(1);
  const [orderPrice, setOrderPrice] = useState<number | undefined>(undefined);
  const [orderProduct, setOrderProduct] = useState("CNC");
  const [placingOrder, setPlacingOrder] = useState(false);

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const res = await indStocksApi.kiteStatus();
      setKiteStatus(res);
      if (res.authenticated) refreshAll();
    } catch { setKiteStatus({ authenticated: false, user_id: null, market_open: false }); }
  };

  const refreshAll = async () => {
    setLoading(true);
    try {
      const [ordersRes, posRes, holdRes] = await Promise.all([
        indStocksApi.orderBook(),
        indStocksApi.positions(),
        indStocksApi.holdings(),
      ]);
      if (ordersRes.success) setOrders(ordersRes.orders);
      if (posRes.success) setPositions({ net: posRes.net, day: posRes.day });
      if (holdRes.success) setHoldings({ holdings: holdRes.holdings, total_pnl: holdRes.total_pnl, day_pnl: holdRes.day_pnl, total_investment: holdRes.total_investment, total_current_value: holdRes.total_current_value });
    } catch (err: any) { toast.error("Failed to refresh data"); }
    finally { setLoading(false); }
  };

  const handlePlaceOrder = async () => {
    if (!orderSymbol.trim()) { toast.error("Enter a symbol"); return; }
    setPlacingOrder(true);
    try {
      const req: PlaceOrderRequest = {
        symbol: orderSymbol.toUpperCase(),
        exchange: "NSE",
        transaction_type: txnType,
        quantity: orderQty,
        order_type: orderType,
        product: orderProduct,
        ...(orderType === "LIMIT" && orderPrice ? { price: orderPrice } : {}),
      };
      const res = await indStocksApi.placeOrder(req);
      if (res.success) {
        toast.success(`Order placed: ${res.order_id}`);
        setOrderSymbol("");
        refreshAll();
      }
    } catch (err: any) { toast.error(err?.message || "Order placement failed"); }
    finally { setPlacingOrder(false); }
  };

  const handleCancelOrder = async (orderId: string) => {
    try {
      const res = await indStocksApi.cancelOrder(orderId);
      if (res.success) { toast.success(`Cancelled: ${orderId}`); refreshAll(); }
    } catch (err: any) { toast.error(err?.message || "Cancel failed"); }
  };

  const orderColumns: ColumnDef<OrderInfo>[] = [
    { accessorKey: "symbol", header: ({ column }) => <SortableHeader column={column} title="Symbol" />, cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span> },
    { accessorKey: "transaction_type", header: "Side", cell: ({ row }) => <Badge variant={row.getValue("transaction_type") === "BUY" ? "default" : "destructive"}>{row.getValue("transaction_type")}</Badge> },
    { accessorKey: "quantity", header: "Qty" },
    { accessorKey: "order_type", header: "Type" },
    { accessorKey: "price", header: "Price", cell: ({ row }) => row.getValue("price") ? `₹${formatNumber(row.getValue("price") as number)}` : "MKT" },
    { accessorKey: "status", header: "Status", cell: ({ row }) => { const s = row.getValue("status") as string; return <Badge variant={s === "COMPLETE" ? "default" : s === "CANCELLED" || s === "REJECTED" ? "destructive" : "outline"}>{s}</Badge>; } },
    { accessorKey: "filled_quantity", header: "Filled" },
    { accessorKey: "average_price", header: "Avg Price", cell: ({ row }) => `₹${formatNumber(row.getValue("average_price") as number)}` },
    { id: "actions", header: "", cell: ({ row }) => row.original.status === "OPEN" || row.original.status === "PENDING" ? <Button size="sm" variant="ghost" onClick={() => handleCancelOrder(row.original.order_id)}><X className="h-3 w-3" /></Button> : null },
  ];

  const positionColumns: ColumnDef<PositionData>[] = [
    { accessorKey: "symbol", header: ({ column }) => <SortableHeader column={column} title="Symbol" />, cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span> },
    { accessorKey: "quantity", header: "Qty", cell: ({ row }) => { const q = row.getValue("quantity") as number; return <span className={q > 0 ? "text-profit" : q < 0 ? "text-loss" : ""}>{q}</span>; } },
    { accessorKey: "average_price", header: "Avg Price", cell: ({ row }) => `₹${formatNumber(row.getValue("average_price") as number)}` },
    { accessorKey: "last_price", header: "LTP", cell: ({ row }) => row.getValue("last_price") ? `₹${formatNumber(row.getValue("last_price") as number)}` : "—" },
    { accessorKey: "pnl", header: ({ column }) => <SortableHeader column={column} title="P&L" />, cell: ({ row }) => { const pnl = row.getValue("pnl") as number | null; return pnl != null ? <span className={pnl >= 0 ? "text-profit font-bold" : "text-loss font-bold"}>₹{formatNumber(pnl)}</span> : "—"; } },
    { accessorKey: "product", header: "Product" },
  ];

  const holdingColumns: ColumnDef<HoldingData>[] = [
    { accessorKey: "symbol", header: ({ column }) => <SortableHeader column={column} title="Symbol" />, cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span> },
    { accessorKey: "quantity", header: "Qty" },
    { accessorKey: "average_price", header: "Avg Price", cell: ({ row }) => `₹${formatNumber(row.getValue("average_price") as number)}` },
    { accessorKey: "last_price", header: "LTP", cell: ({ row }) => row.getValue("last_price") ? `₹${formatNumber(row.getValue("last_price") as number)}` : "—" },
    { accessorKey: "pnl", header: ({ column }) => <SortableHeader column={column} title="P&L" />, cell: ({ row }) => { const pnl = row.getValue("pnl") as number | null; return pnl != null ? <span className={pnl >= 0 ? "text-profit font-bold" : "text-loss font-bold"}>₹{formatNumber(pnl)}</span> : "—"; } },
    { accessorKey: "day_change_pct", header: "Day %", cell: ({ row }) => { const d = row.getValue("day_change_pct") as number | null; return d != null ? <span className={d >= 0 ? "text-profit" : "text-loss"}>{d >= 0 ? "+" : ""}{formatNumber(d, 2)}%</span> : "—"; } },
  ];

  const totalPositionPnl = positions.net.reduce((sum, p) => sum + (p.pnl || 0), 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Kite Trading Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            {kiteStatus?.authenticated ? (
              <span className="text-profit">Connected — {kiteStatus.user_id} {kiteStatus.market_open ? "(Market Open)" : "(Market Closed)"}</span>
            ) : (
              <span className="text-loss">Not connected to Kite</span>
            )}
          </p>
        </div>
        <Button onClick={refreshAll} disabled={loading} variant="outline" size="sm">
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />Refresh
        </Button>
      </div>

      {/* Quick Order Form */}
      <Card>
        <CardHeader><CardTitle className="text-base">Place Order</CardTitle></CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            <div className="w-32"><Label className="text-xs">Symbol</Label><Input value={orderSymbol} onChange={(e) => setOrderSymbol(e.target.value)} placeholder="RELIANCE" /></div>
            <div className="w-24">
              <Label className="text-xs">Side</Label>
              <Select value={txnType} onValueChange={setTxnType}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="BUY">BUY</SelectItem><SelectItem value="SELL">SELL</SelectItem></SelectContent></Select>
            </div>
            <div className="w-20"><Label className="text-xs">Qty</Label><Input type="number" value={orderQty} onChange={(e) => setOrderQty(Number(e.target.value))} min={1} /></div>
            <div className="w-28">
              <Label className="text-xs">Type</Label>
              <Select value={orderType} onValueChange={setOrderType}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="MARKET">Market</SelectItem><SelectItem value="LIMIT">Limit</SelectItem><SelectItem value="SL">SL</SelectItem><SelectItem value="SL-M">SL-M</SelectItem></SelectContent></Select>
            </div>
            {orderType === "LIMIT" && (
              <div className="w-28"><Label className="text-xs">Price</Label><Input type="number" value={orderPrice || ""} onChange={(e) => setOrderPrice(Number(e.target.value))} step={0.05} /></div>
            )}
            <div className="w-24">
              <Label className="text-xs">Product</Label>
              <Select value={orderProduct} onValueChange={setOrderProduct}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="CNC">CNC</SelectItem><SelectItem value="MIS">MIS</SelectItem><SelectItem value="NRML">NRML</SelectItem></SelectContent></Select>
            </div>
            <Button onClick={handlePlaceOrder} disabled={placingOrder || !kiteStatus?.authenticated}>
              {placingOrder ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShoppingCart className="mr-2 h-4 w-4" />}
              Place
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Summary Metrics */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        <MetricCard title="Total Investment" value={`₹${formatNumber(holdings.total_investment)}`} icon={<Briefcase className="h-4 w-4" />} />
        <MetricCard title="Current Value" value={`₹${formatNumber(holdings.total_current_value)}`} />
        <MetricCard title="Holdings P&L" value={`₹${formatNumber(holdings.total_pnl)}`} changeType={holdings.total_pnl >= 0 ? "positive" : "negative"} />
        <MetricCard title="Day P&L" value={`₹${formatNumber(holdings.day_pnl)}`} changeType={holdings.day_pnl >= 0 ? "positive" : "negative"} />
        <MetricCard title="Position P&L" value={`₹${formatNumber(totalPositionPnl)}`} changeType={totalPositionPnl >= 0 ? "positive" : "negative"} />
      </div>

      {/* Data Tabs */}
      <Tabs defaultValue="orders">
        <TabsList>
          <TabsTrigger value="orders"><ListOrdered className="mr-1 h-3 w-3" />Orders ({orders.length})</TabsTrigger>
          <TabsTrigger value="positions"><Package className="mr-1 h-3 w-3" />Positions ({positions.net.length})</TabsTrigger>
          <TabsTrigger value="holdings"><Briefcase className="mr-1 h-3 w-3" />Holdings ({holdings.holdings.length})</TabsTrigger>
        </TabsList>
        <TabsContent value="orders">
          <Card><CardContent className="pt-6"><DataTable columns={orderColumns} data={orders} searchKey="symbol" /></CardContent></Card>
        </TabsContent>
        <TabsContent value="positions">
          <Card><CardContent className="pt-6"><DataTable columns={positionColumns} data={positions.net} searchKey="symbol" /></CardContent></Card>
        </TabsContent>
        <TabsContent value="holdings">
          <Card><CardContent className="pt-6"><DataTable columns={holdingColumns} data={holdings.holdings} searchKey="symbol" /></CardContent></Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
