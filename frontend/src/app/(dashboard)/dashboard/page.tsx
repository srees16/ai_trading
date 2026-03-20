"use client";

import { useEffect, useState } from "react";
import { healthApi } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { MetricCard } from "@/components/shared/metric-card";
import { FearGreedGauge } from "@/components/shared/fear-greed-gauge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Activity, Database, Server, Wifi, TrendingUp, Shield, BarChart3, Zap } from "lucide-react";

interface SystemHealth {
  status: string;
  database: boolean;
  version: string;
  components: Record<string, boolean>;
}

export default function DashboardPage() {
  const market = useAppStore((s) => s.market);
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    healthApi
      .check()
      .then(setHealth)
      .catch(() => {});
  }, []);

  const components = health?.components || {};

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          System overview and {market === "US" ? "US Markets" : "Indian Markets"} status
        </p>
      </div>

      {/* System Health Cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="System Status"
          value={health ? (health.status === "healthy" ? "Operational" : "Degraded") : "—"}
          change={`v${health?.version || "—"}`}
          changeType={health?.status === "healthy" ? "positive" : "neutral"}
          icon={<Server className="h-4 w-4" />}
        />
        <MetricCard
          title="Database"
          value={health ? (health.database ? "Connected" : "Offline") : "—"}
          changeType={health?.database ? "positive" : "neutral"}
          icon={<Database className="h-4 w-4" />}
        />
        <MetricCard
          title="Active Components"
          value={`${Object.values(components).filter(Boolean).length}/${Object.keys(components).length}`}
          changeType="neutral"
          icon={<Activity className="h-4 w-4" />}
        />
        <MetricCard
          title="Market"
          value={market === "US" ? "US Markets" : "Indian Markets"}
          changeType="neutral"
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>

      {/* Components Grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="h-4 w-4 text-primary" />
              Infrastructure Components
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(components).map(([name, status]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-sm capitalize">{name.replace(/_/g, " ")}</span>
                  <Badge variant={status ? "default" : "destructive"} className="text-xs">
                    {status ? "Online" : "Offline"}
                  </Badge>
                </div>
              ))}
              {Object.keys(components).length === 0 && (
                <p className="text-sm text-muted-foreground">No component data available</p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Shield className="h-4 w-4 text-primary" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {market === "US" ? (
                <>
                  <ActionCard title="Run Analysis" description="US stock signals" href="/us/analysis" icon={<BarChart3 className="h-5 w-5" />} />
                  <ActionCard title="Backtest" description="Strategy testing" href="/us/backtesting" icon={<TrendingUp className="h-5 w-5" />} />
                  <ActionCard title="Verdict" description="Integrated scorer" href="/us/verdict" icon={<Shield className="h-5 w-5" />} />
                  <ActionCard title="Holdings" description="DriveWealth" href="/us/holdings" icon={<Activity className="h-5 w-5" />} />
                </>
              ) : (
                <>
                  <ActionCard title="Analysis" description="NSE stock signals" href="/ind/analysis" icon={<BarChart3 className="h-5 w-5" />} />
                  <ActionCard title="Screener" description="NSE screener" href="/ind/screener" icon={<Wifi className="h-5 w-5" />} />
                  <ActionCard title="Options" description="Option chain" href="/ind/options" icon={<TrendingUp className="h-5 w-5" />} />
                  <ActionCard title="Trading" description="Kite orders" href="/ind/fly-kite" icon={<Activity className="h-5 w-5" />} />
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {market === "IND" && <FearGreedGauge />}
      </div>
    </div>
  );
}

function ActionCard({
  title,
  description,
  href,
  icon,
}: {
  title: string;
  description: string;
  href: string;
  icon: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className="flex items-center gap-3 rounded-lg border border-border/50 p-3 transition-colors hover:bg-accent"
    >
      <div className="text-primary">{icon}</div>
      <div>
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
    </Link>
  );
}
