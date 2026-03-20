"use client";

import { DriveWealthAuth } from "@/components/forms/drivewealth-auth";
import { OrderForm } from "@/components/forms/order-form";
import { MetricsGrid, MetricCard } from "@/components/common/metrics-cards";
import { RibbonVixBar } from "@/components/common/ribbon-vix-bar";
import { Spinner } from "@/components/common/spinner";
import { NASDAQ_50_TICKERS } from "@/lib/constants";
import { useDriveWealth } from "@/hooks/use-drivewealth";
import { formatCurrency } from "@/lib/utils";

export default function USHoldingsPage() {
  const {
    login, isLoggingIn, loginError, account, isConnected,
    positions, fetchPositions, placeOrder, isOrdering, orderError, logout,
  } = useDriveWealth();

  const totalPnl = positions.reduce((a, p) => a + p.unrealized_pnl, 0);

  return (
    <div className="space-y-6">
      <RibbonVixBar symbols={NASDAQ_50_TICKERS} market="US" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="space-y-4">
          <DriveWealthAuth
            account={account}
            isConnected={isConnected}
            isLoading={isLoggingIn}
            error={loginError}
            onLogin={login}
            onLogout={logout}
          />

          {isConnected && (
            <OrderForm
              tickers={positions.map((p) => p.symbol)}
              onSubmit={(o) => placeOrder({ symbol: o.ticker, side: o.side, order_type: o.order_type, quantity: o.quantity, limit_price: o.limit_price })}
              isSubmitting={isOrdering}
              error={orderError}
            />
          )}
        </div>

        <div className="md:col-span-3 space-y-4">
          {isConnected && account && (
            <MetricsGrid>
              <MetricCard label="Cash Balance" value={formatCurrency(account.cash_balance)} />
              <MetricCard label="Equity Value" value={formatCurrency(account.equity_value)} />
              <MetricCard label="Available" value={formatCurrency(account.available_for_trading)} />
              <MetricCard label="Unrealized P&L" value={formatCurrency(totalPnl)} color={totalPnl >= 0 ? "text-green-500" : "text-red-500"} />
            </MetricsGrid>
          )}

          {isConnected && (
            <div className="content-panel p-4">
              <h4 className="text-sm font-semibold mb-2">Positions</h4>
              {positions.length === 0 ? (
                <p className="text-sm text-muted-foreground">No open positions</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-muted-foreground text-left">
                        <th className="py-2 pr-4">Symbol</th>
                        <th className="py-2 pr-4">Qty</th>
                        <th className="py-2 pr-4">Avg Price</th>
                        <th className="py-2 pr-4">Mkt Value</th>
                        <th className="py-2 pr-4">P&L</th>
                        <th className="py-2">P&L %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((p) => (
                        <tr key={p.symbol} className="border-b">
                          <td className="py-2 pr-4 font-mono font-medium">{p.symbol}</td>
                          <td className="py-2 pr-4">{p.quantity}</td>
                          <td className="py-2 pr-4">{formatCurrency(p.avg_price)}</td>
                          <td className="py-2 pr-4">{formatCurrency(p.market_value)}</td>
                          <td className={`py-2 pr-4 ${p.unrealized_pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                            {formatCurrency(p.unrealized_pnl)}
                          </td>
                          <td className={`py-2 ${p.unrealized_pnl_pct >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                            {(p.unrealized_pnl_pct * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
