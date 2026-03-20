"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { AlertCircle, ShoppingCart } from "lucide-react";
import { ConfirmationDialog } from "@/components/common/confirmation-dialog";

interface OrderFormProps {
  tickers: string[];
  onSubmit: (order: {
    ticker: string;
    side: "buy" | "sell";
    order_type: "market" | "limit";
    quantity: number;
    limit_price?: number;
  }) => void;
  isSubmitting: boolean;
  error: string | null;
}

export function OrderForm({ tickers, onSubmit, isSubmitting, error }: OrderFormProps) {
  const [ticker, setTicker] = useState(tickers[0] || "");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [orderType, setOrderType] = useState<"market" | "limit">("market");
  const [quantity, setQuantity] = useState(1);
  const [limitPrice, setLimitPrice] = useState<number>(0);
  const [showConfirm, setShowConfirm] = useState(false);

  const tickerValid = /^[A-Z]{1,10}$/.test(ticker);
  const quantityValid = Number.isInteger(quantity) && quantity >= 1 && quantity <= 100_000;
  const limitPriceValid = orderType !== "limit" || (limitPrice > 0 && limitPrice <= 1_000_000);
  const formValid = tickerValid && quantityValid && limitPriceValid;

  const handleSubmit = () => {
    if (!formValid) return;
    const order: Parameters<typeof onSubmit>[0] = {
      ticker: ticker.toUpperCase(),
      side,
      order_type: orderType,
      quantity: Math.floor(quantity),
    };
    if (orderType === "limit" && limitPrice > 0) order.limit_price = limitPrice;
    onSubmit(order);
    setShowConfirm(false);
  };

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <p className="text-sm font-medium flex items-center gap-1">
        <ShoppingCart className="h-4 w-4" /> Place Order
      </p>

      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" /> {error}
        </div>
      )}

      {/* Ticker */}
      <div>
        <Label className="text-xs">Ticker</Label>
        {tickers.length > 0 ? (
          <Select value={ticker} onValueChange={setTicker}>
            <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              {tickers.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
            </SelectContent>
          </Select>
        ) : (
          <Input
            className="h-8 text-sm font-mono"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.replace(/[^A-Za-z]/g, "").toUpperCase().slice(0, 10))}
            placeholder="AAPL"
          />
        )}
      </div>

      {/* Side + Type */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">Side</Label>
          <Select value={side} onValueChange={(v) => setSide(v as "buy" | "sell")}>
            <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="buy">Buy</SelectItem>
              <SelectItem value="sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs">Order Type</Label>
          <Select value={orderType} onValueChange={(v) => setOrderType(v as "market" | "limit")}>
            <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="market">Market</SelectItem>
              <SelectItem value="limit">Limit</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Quantity */}
      <div>
        <Label className="text-xs">Quantity</Label>
        <Input type="number" className="h-8 text-sm" value={quantity} min={1} step={1} onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))} />
      </div>

      {/* Limit Price */}
      {orderType === "limit" && (
        <div>
          <Label className="text-xs">Limit Price ($)</Label>
          <Input type="number" className="h-8 text-sm" value={limitPrice} min={0.01} step={0.01} onChange={(e) => setLimitPrice(Number(e.target.value))} />
        </div>
      )}

      <Button
        className="w-full h-8 text-sm"
        variant={side === "buy" ? "success" : "destructive"}
        disabled={isSubmitting || !formValid}
        onClick={() => setShowConfirm(true)}
      >
        {isSubmitting ? "Submitting…" : `${side === "buy" ? "Buy" : "Sell"} ${ticker}`}
      </Button>

      <ConfirmationDialog
        open={showConfirm}
        title="Confirm Order"
        description={`${side.toUpperCase()} ${quantity} share(s) of ${ticker} (${orderType}${orderType === "limit" ? ` @ $${limitPrice}` : ""})`}
        confirmLabel="Place Order"
        onConfirm={handleSubmit}
        onCancel={() => setShowConfirm(false)}
      />
    </div>
  );
}
