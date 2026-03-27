"use client";

import { useState, useEffect } from "react";
import { healthApi } from "@/lib/api";
import { useAuthStore, useThemeStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { MetricCard } from "@/components/shared/metric-card";
import { RefreshCw, Server, Shield, Globe, Loader2, Sun, Moon, Palette } from "lucide-react";
import { toast } from "sonner";

export default function SettingsPage() {
  const { username, role, loginTime } = useAuthStore();
  const { theme, setTheme } = useThemeStore();
  const [health, setHealth] = useState<{
    status: string;
    database: boolean;
    version: string;
    timestamp: string;
    components: Record<string, boolean>;
  } | null>(null);
  const [infra, setInfra] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshHealth = async () => {
    setLoading(true);
    try {
      const [h, i] = await Promise.all([healthApi.check(), healthApi.infra()]);
      setHealth(h);
      setInfra(i);
    } catch (err: any) { toast.error("Failed to fetch system status"); }
    finally { setLoading(false); }
  };

  useEffect(() => { refreshHealth(); }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground">System configuration and status</p>
        </div>
        <Button onClick={refreshHealth} disabled={loading} variant="outline" size="sm">
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
          Refresh
        </Button>
      </div>

      {/* Appearance */}
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Palette className="h-4 w-4" />Appearance</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label className="text-sm text-muted-foreground mb-3 block">Theme</Label>
            <div className="flex gap-3">
              <Button
                variant={theme === "light" ? "default" : "outline"}
                size="sm"
                onClick={() => setTheme("light")}
                className="flex items-center gap-2"
              >
                <Sun className="h-4 w-4" />
                Light
              </Button>
              <Button
                variant={theme === "dark" ? "default" : "outline"}
                size="sm"
                onClick={() => setTheme("dark")}
                className="flex items-center gap-2"
              >
                <Moon className="h-4 w-4" />
                Dark
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* User Info */}
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Shield className="h-4 w-4" />User Session</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
            <div><span className="text-muted-foreground">Username:</span> <span className="font-bold">{username || "—"}</span></div>
            <div><span className="text-muted-foreground">Role:</span> <Badge variant="outline">{role || "user"}</Badge></div>
            <div><span className="text-muted-foreground">Login Time:</span> <span className="font-mono text-xs">{loginTime ? new Date(loginTime).toLocaleString() : "—"}</span></div>
            <div><span className="text-muted-foreground">Status:</span> <Badge className="bg-profit/20 text-profit">Active</Badge></div>
          </div>
        </CardContent>
      </Card>

      {/* System Health */}
      {health && (
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Server className="h-4 w-4" />System Health</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
              <div><span className="text-muted-foreground">Status:</span> <Badge className={health.status === "healthy" ? "bg-profit/20 text-profit" : "bg-loss/20 text-loss"}>{health.status}</Badge></div>
              <div><span className="text-muted-foreground">Version:</span> <span className="font-mono">{health.version}</span></div>
              <div><span className="text-muted-foreground">Database:</span> <Badge className={health.database ? "bg-profit/20 text-profit" : "bg-loss/20 text-loss"}>{health.database ? "Connected" : "Disconnected"}</Badge></div>
              <div><span className="text-muted-foreground">Timestamp:</span> <span className="font-mono text-xs">{health.timestamp}</span></div>
            </div>
            <Separator />
            <div>
              <p className="text-xs text-muted-foreground mb-2">Components:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(health.components).map(([name, ok]) => (
                  <Badge key={name} variant="outline" className={ok ? "border-profit/50 text-profit" : "border-loss/50 text-loss"}>
                    {name}: {ok ? "OK" : "DOWN"}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Infrastructure */}
      {infra && (
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Globe className="h-4 w-4" />Infrastructure</CardTitle></CardHeader>
          <CardContent>
            <pre className="rounded bg-muted p-4 text-xs overflow-auto max-h-64">{JSON.stringify(infra, null, 2)}</pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
