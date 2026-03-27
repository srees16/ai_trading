"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MetricCard } from "@/components/shared/metric-card";
import { DataTable, SortableHeader } from "@/components/shared/data-table";
import { formatCurrency, formatNumber, formatPercent, getPnlColor } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import { Loader2, Link2, Unlink, Briefcase, DollarSign } from "lucide-react";
import { toast } from "sonner";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9001";

interface DWPosition {
  symbol: string;
  quantity: number;
  averagePrice: number;
  marketValue: number;
  unrealizedPL: number;
  side: string;
}

export default function USHoldingsPage() {
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [appKey, setAppKey] = useState("");
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [accountData, setAccountData] = useState<Record<string, unknown> | null>(null);
  const [positions, setPositions] = useState<DWPosition[]>([]);

  const handleConnect = async () => {
    if (!clientId || !clientSecret || !appKey) {
      toast.error("Enter all DriveWealth credentials");
      return;
    }
    setLoading(true);
    try {
      // DriveWealth integration through backend
      toast.info("DriveWealth connection — use the backend API for full integration");
      setConnected(true);
    } catch {
      toast.error("Connection failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = () => {
    setConnected(false);
    setAccountData(null);
    setPositions([]);
    toast.info("Disconnected");
  };

  const positionColumns: ColumnDef<DWPosition>[] = [
    {
      accessorKey: "symbol",
      header: ({ column }) => <SortableHeader column={column} title="Symbol" />,
      cell: ({ row }) => <span className="font-mono font-bold">{row.getValue("symbol")}</span>,
    },
    { accessorKey: "quantity", header: "Qty" },
    {
      accessorKey: "averagePrice",
      header: "Avg Price",
      cell: ({ row }) => formatCurrency(row.getValue("averagePrice") as number),
    },
    {
      accessorKey: "marketValue",
      header: "Market Value",
      cell: ({ row }) => formatCurrency(row.getValue("marketValue") as number),
    },
    {
      accessorKey: "unrealizedPL",
      header: ({ column }) => <SortableHeader column={column} title="P&L" />,
      cell: ({ row }) => {
        const v = row.getValue("unrealizedPL") as number;
        return <span className={getPnlColor(v)}>{formatCurrency(v)}</span>;
      },
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">US Holdings</h1>
        <p className="text-sm text-muted-foreground">DriveWealth portfolio management</p>
      </div>

      {!connected ? (
        <Card className="max-w-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Link2 className="h-4 w-4" />
              Connect DriveWealth
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Client ID</Label>
              <Input type="password" value={clientId} onChange={(e) => setClientId(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Client Secret</Label>
              <Input type="password" value={clientSecret} onChange={(e) => setClientSecret(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>App Key</Label>
              <Input type="password" value={appKey} onChange={(e) => setAppKey(e.target.value)} />
            </div>
            <Button onClick={handleConnect} disabled={loading} className="w-full">
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Link2 className="mr-2 h-4 w-4" />}
              Connect
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="flex items-center gap-3">
            <span className="text-sm text-profit">Connected to DriveWealth</span>
            <Button variant="outline" size="sm" onClick={handleDisconnect}>
              <Unlink className="mr-1.5 h-3.5 w-3.5" /> Disconnect
            </Button>
          </div>

          <Tabs defaultValue="summary">
            <TabsList>
              <TabsTrigger value="summary">Account Summary</TabsTrigger>
              <TabsTrigger value="positions">Positions</TabsTrigger>
              <TabsTrigger value="orders">Orders</TabsTrigger>
            </TabsList>

            <TabsContent value="summary">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <MetricCard title="Portfolio Value" value="—" icon={<Briefcase className="h-4 w-4" />} />
                <MetricCard title="Cash Balance" value="—" icon={<DollarSign className="h-4 w-4" />} />
                <MetricCard title="Day P&L" value="—" />
                <MetricCard title="Total P&L" value="—" />
              </div>
            </TabsContent>

            <TabsContent value="positions">
              <Card>
                <CardContent className="pt-6">
                  <DataTable columns={positionColumns} data={positions} searchKey="symbol" />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="orders">
              <Card>
                <CardContent className="pt-6">
                  <p className="text-center text-muted-foreground">Order history will appear here</p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
