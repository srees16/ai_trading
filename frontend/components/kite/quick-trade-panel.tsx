"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { useKitePlaceOrder } from "@/hooks/use-kite";
import { Zap, FileText, Shield, Clock } from "lucide-react";

const EXCHANGES = ["NSE", "BSE"];
const PRODUCTS = ["CNC", "MIS", "NRML"];
const ORDER_TYPES = ["LIMIT", "MARKET", "SL", "SL-M"];

type KiteOrderPayload = {
  tradingsymbol: string;
  exchange: string;
  transaction_type: "BUY" | "SELL";
  order_type: "MARKET" | "LIMIT" | "SL" | "SL-M";
  quantity: number;
  price?: number;
  trigger_price?: number;
  product?: string;
  validity?: string;
  variety?: string;
};

interface QuickTradePanelProps {
  symbols: string[];
}

export function QuickTradePanel({ symbols }: QuickTradePanelProps) {
  const [tab, setTab] = useState("quick");
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const placeOrder = useKitePlaceOrder();

  const sorted = [...symbols].sort();

  const handleResult = (txn: string, result: { order_id: string }) => {
    setMsg({ type: "success", text: `${txn} order placed — ID: ${result.order_id}` });
  };

  const handleError = (err: Error) => {
    setMsg({ type: "error", text: err.message });
  };

  return (
    <div className="content-panel p-4 space-y-3">
      <h3 className="text-sm font-semibold">Quick Trade</h3>
      {msg && (
        <p className={`text-xs px-2 py-1 rounded ${msg.type === "success" ? "bg-green-500/10 text-green-600" : "bg-red-500/10 text-red-600"}`}>
          {msg.text}
        </p>
      )}
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="grid grid-cols-4 h-8">
          <TabsTrigger value="quick" className="text-xs gap-1">
            <Zap className="h-3 w-3" /> Quick
          </TabsTrigger>
          <TabsTrigger value="regular" className="text-xs gap-1">
            <FileText className="h-3 w-3" /> Regular
          </TabsTrigger>
          <TabsTrigger value="cover" className="text-xs gap-1">
            <Shield className="h-3 w-3" /> Cover
          </TabsTrigger>
          <TabsTrigger value="amo" className="text-xs gap-1">
            <Clock className="h-3 w-3" /> AMO
          </TabsTrigger>
        </TabsList>

        {/* ── Quick ── */}
        <TabsContent value="quick" className="mt-3 space-y-2">
          <QuickTab
            symbols={sorted}
            onPlace={(o) =>
              placeOrder.mutate(o, {
                onSuccess: (r) => handleResult(o.transaction_type, r),
                onError: handleError,
              })
            }
            isPending={placeOrder.isPending}
          />
        </TabsContent>

        {/* ── Regular ── */}
        <TabsContent value="regular" className="mt-3 space-y-2">
          <RegularTab
            symbols={sorted}
            onPlace={(o) =>
              placeOrder.mutate(o, {
                onSuccess: (r) => handleResult(o.transaction_type, r),
                onError: handleError,
              })
            }
            isPending={placeOrder.isPending}
          />
        </TabsContent>

        {/* ── Cover Order ── */}
        <TabsContent value="cover" className="mt-3 space-y-2">
          <CoverTab
            symbols={sorted}
            onPlace={(o) =>
              placeOrder.mutate(o, {
                onSuccess: (r) => handleResult(o.transaction_type, r),
                onError: handleError,
              })
            }
            isPending={placeOrder.isPending}
          />
        </TabsContent>

        {/* ── AMO ── */}
        <TabsContent value="amo" className="mt-3 space-y-2">
          <AmoTab
            symbols={sorted}
            onPlace={(o) =>
              placeOrder.mutate(o, {
                onSuccess: (r) => handleResult(o.transaction_type, r),
                onError: handleError,
              })
            }
            isPending={placeOrder.isPending}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   Quick Tab — Market order, minimal inputs
   ═══════════════════════════════════════════════════════════ */
interface TabProps {
  symbols: string[];
  onPlace: (o: KiteOrderPayload) => void;
  isPending: boolean;
}

function QuickTab({ symbols, onPlace, isPending }: TabProps) {
  const [sym, setSym] = useState(symbols[0] ?? "");
  const [qty, setQty] = useState(1);
  const [product, setProduct] = useState("CNC");
  const [exchange, setExchange] = useState("NSE");

  const place = (side: "BUY" | "SELL") =>
    onPlace({
      tradingsymbol: sym,
      exchange,
      transaction_type: side,
      quantity: qty,
      order_type: "MARKET",
      product,
    });

  return (
    <>
      <p className="text-xs text-muted-foreground">Market order — instant execution</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-xs">Symbol</Label>
          <Select value={sym} onValueChange={setSym}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{symbols.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Qty</Label>
          <Input type="number" className="h-8 text-xs" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value) || 1)} />
        </div>
        <div>
          <Label className="text-xs">Product</Label>
          <Select value={product} onValueChange={setProduct}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{PRODUCTS.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Exchange</Label>
          <Select value={exchange} onValueChange={setExchange}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{EXCHANGES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white" onClick={() => place("BUY")} disabled={isPending}>
          BUY
        </Button>
        <Button size="sm" variant="outline" className="border-red-400 text-red-500 hover:bg-red-50" onClick={() => place("SELL")} disabled={isPending}>
          SELL
        </Button>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════
   Regular Tab — Full-featured order
   ═══════════════════════════════════════════════════════════ */
function RegularTab({ symbols, onPlace, isPending }: TabProps) {
  const [sym, setSym] = useState(symbols[0] ?? "");
  const [exchange, setExchange] = useState("NSE");
  const [orderType, setOrderType] = useState("LIMIT");
  const [product, setProduct] = useState("CNC");
  const [qty, setQty] = useState(1);
  const [price, setPrice] = useState(0);
  const [trigger, setTrigger] = useState(0);
  const [validity, setValidity] = useState("DAY");
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");

  const place = () => {
    const o: KiteOrderPayload = {
      tradingsymbol: sym,
      exchange,
      transaction_type: side,
      quantity: qty,
      order_type: orderType as KiteOrderPayload["order_type"],
      product,
      validity,
    };
    if (orderType !== "MARKET" && price > 0) o.price = price;
    if ((orderType === "SL" || orderType === "SL-M") && trigger > 0) o.trigger_price = trigger;
    onPlace(o);
  };

  return (
    <>
      <p className="text-xs text-muted-foreground">Limit / SL / SL-M with all parameters</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-xs">Symbol</Label>
          <Select value={sym} onValueChange={setSym}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{symbols.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Exchange</Label>
          <Select value={exchange} onValueChange={setExchange}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{EXCHANGES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Order Type</Label>
          <Select value={orderType} onValueChange={setOrderType}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{ORDER_TYPES.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Product</Label>
          <Select value={product} onValueChange={setProduct}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{PRODUCTS.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Qty</Label>
          <Input type="number" className="h-8 text-xs" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value) || 1)} />
        </div>
        <div>
          <Label className="text-xs">Price</Label>
          <Input type="number" className="h-8 text-xs" min={0} step={0.05} value={price} onChange={(e) => setPrice(Number(e.target.value))} disabled={orderType === "MARKET"} />
        </div>
        <div>
          <Label className="text-xs">Trigger Price</Label>
          <Input type="number" className="h-8 text-xs" min={0} step={0.05} value={trigger} onChange={(e) => setTrigger(Number(e.target.value))} disabled={orderType !== "SL" && orderType !== "SL-M"} />
        </div>
        <div>
          <Label className="text-xs">Validity</Label>
          <Select value={validity} onValueChange={setValidity}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="DAY">DAY</SelectItem>
              <SelectItem value="IOC">IOC</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <div className="flex gap-2">
          <Button size="sm" variant={side === "BUY" ? "default" : "outline"} className={side === "BUY" ? "bg-green-600 hover:bg-green-700" : ""} onClick={() => setSide("BUY")}>
            BUY
          </Button>
          <Button size="sm" variant={side === "SELL" ? "default" : "outline"} className={side === "SELL" ? "bg-red-600 hover:bg-red-700" : ""} onClick={() => setSide("SELL")}>
            SELL
          </Button>
        </div>
        <Button size="sm" className="flex-1" onClick={place} disabled={isPending}>
          Place {side} Order
        </Button>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════
   Cover Order Tab — Intraday with built-in SL
   ═══════════════════════════════════════════════════════════ */
function CoverTab({ symbols, onPlace, isPending }: TabProps) {
  const [sym, setSym] = useState(symbols[0] ?? "");
  const [exchange, setExchange] = useState("NSE");
  const [orderType, setOrderType] = useState("MARKET");
  const [qty, setQty] = useState(1);
  const [price, setPrice] = useState(0);
  const [trigger, setTrigger] = useState(1);
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");

  const place = () => {
    const o: KiteOrderPayload = {
      tradingsymbol: sym,
      exchange,
      transaction_type: side,
      quantity: qty,
      order_type: orderType as KiteOrderPayload["order_type"],
      product: "MIS",
      validity: "DAY",
      variety: "co",
      trigger_price: trigger,
    };
    if (orderType === "LIMIT" && price > 0) o.price = price;
    onPlace(o);
  };

  return (
    <>
      <p className="text-xs text-muted-foreground">Intraday with built-in stop-loss (MIS only)</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-xs">Symbol</Label>
          <Select value={sym} onValueChange={setSym}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{symbols.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Exchange</Label>
          <Select value={exchange} onValueChange={setExchange}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{EXCHANGES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Order Type</Label>
          <Select value={orderType} onValueChange={setOrderType}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="MARKET">MARKET</SelectItem>
              <SelectItem value="LIMIT">LIMIT</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Qty</Label>
          <Input type="number" className="h-8 text-xs" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value) || 1)} />
        </div>
        <div>
          <Label className="text-xs">Price</Label>
          <Input type="number" className="h-8 text-xs" min={0} step={0.05} value={price} onChange={(e) => setPrice(Number(e.target.value))} disabled={orderType === "MARKET"} />
        </div>
        <div>
          <Label className="text-xs">SL Trigger *</Label>
          <Input type="number" className="h-8 text-xs" min={0.05} step={0.05} value={trigger} onChange={(e) => setTrigger(Number(e.target.value) || 0.05)} />
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <div className="flex gap-2">
          <Button size="sm" variant={side === "BUY" ? "default" : "outline"} className={side === "BUY" ? "bg-green-600 hover:bg-green-700" : ""} onClick={() => setSide("BUY")}>
            BUY
          </Button>
          <Button size="sm" variant={side === "SELL" ? "default" : "outline"} className={side === "SELL" ? "bg-red-600 hover:bg-red-700" : ""} onClick={() => setSide("SELL")}>
            SELL
          </Button>
        </div>
        <Button size="sm" className="flex-1" onClick={place} disabled={isPending}>
          Place Cover {side}
        </Button>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════
   AMO Tab — After Market Order
   ═══════════════════════════════════════════════════════════ */
function AmoTab({ symbols, onPlace, isPending }: TabProps) {
  const [sym, setSym] = useState(symbols[0] ?? "");
  const [exchange, setExchange] = useState("NSE");
  const [orderType, setOrderType] = useState("LIMIT");
  const [product, setProduct] = useState("CNC");
  const [qty, setQty] = useState(1);
  const [price, setPrice] = useState(0);
  const [trigger, setTrigger] = useState(0);
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");

  const place = () => {
    const o: KiteOrderPayload = {
      tradingsymbol: sym,
      exchange,
      transaction_type: side,
      quantity: qty,
      order_type: orderType as KiteOrderPayload["order_type"],
      product,
      validity: "DAY",
      variety: "amo",
    };
    if (orderType !== "MARKET" && price > 0) o.price = price;
    if ((orderType === "SL" || orderType === "SL-M") && trigger > 0) o.trigger_price = trigger;
    onPlace(o);
  };

  return (
    <>
      <p className="text-xs text-muted-foreground">After-market order — queued for next session</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-xs">Symbol</Label>
          <Select value={sym} onValueChange={setSym}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{symbols.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Exchange</Label>
          <Select value={exchange} onValueChange={setExchange}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{EXCHANGES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Order Type</Label>
          <Select value={orderType} onValueChange={setOrderType}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{ORDER_TYPES.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Product</Label>
          <Select value={product} onValueChange={setProduct}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>{PRODUCTS.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Qty</Label>
          <Input type="number" className="h-8 text-xs" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value) || 1)} />
        </div>
        <div>
          <Label className="text-xs">Price</Label>
          <Input type="number" className="h-8 text-xs" min={0} step={0.05} value={price} onChange={(e) => setPrice(Number(e.target.value))} disabled={orderType === "MARKET"} />
        </div>
        <div>
          <Label className="text-xs">Trigger Price</Label>
          <Input type="number" className="h-8 text-xs" min={0} step={0.05} value={trigger} onChange={(e) => setTrigger(Number(e.target.value))} disabled={orderType !== "SL" && orderType !== "SL-M"} />
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <div className="flex gap-2">
          <Button size="sm" variant={side === "BUY" ? "default" : "outline"} className={side === "BUY" ? "bg-green-600 hover:bg-green-700" : ""} onClick={() => setSide("BUY")}>
            BUY
          </Button>
          <Button size="sm" variant={side === "SELL" ? "default" : "outline"} className={side === "SELL" ? "bg-red-600 hover:bg-red-700" : ""} onClick={() => setSide("SELL")}>
            SELL
          </Button>
        </div>
        <Button size="sm" className="flex-1" onClick={place} disabled={isPending}>
          Place AMO {side}
        </Button>
      </div>
    </>
  );
}
