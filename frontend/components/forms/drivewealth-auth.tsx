"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Eye, EyeOff, LogIn, LogOut, AlertCircle } from "lucide-react";
import type { DWCredentials, DWAccount } from "@/lib/types";

interface DriveWealthAuthProps {
  account: DWAccount | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  onLogin: (creds: DWCredentials) => void;
  onLogout: () => void;
}

export function DriveWealthAuth({ account, isConnected, isLoading, error, onLogin, onLogout }: DriveWealthAuthProps) {
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [appKey, setAppKey] = useState("");
  const [userId, setUserId] = useState("");
  const [accountId, setAccountId] = useState("");
  const [showPw, setShowPw] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (clientId.trim() && clientSecret.trim() && appKey.trim() && userId.trim() && accountId.trim()) {
      onLogin({
        client_id: clientId.trim(),
        client_secret: clientSecret.trim(),
        app_key: appKey.trim(),
        user_id: userId.trim(),
        account_id: accountId.trim(),
      });
    }
  };

  if (isConnected && account) {
    return (
      <div className="rounded-lg border bg-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-green-600">● Connected</span>
          <Button variant="outline" size="sm" onClick={onLogout}>
            <LogOut className="mr-1 h-3 w-3" /> Disconnect
          </Button>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-muted-foreground">Account:</span>{" "}
            <span className="font-mono">{account.account_no}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Status:</span>{" "}
            <span className="capitalize">{account.status}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Cash:</span>{" "}
            <span className="font-mono">${account.cash_balance?.toLocaleString() ?? "—"}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Equity:</span>{" "}
            <span className="font-mono">${account.equity_value?.toLocaleString() ?? "—"}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-lg border bg-card p-4 space-y-3">
      <p className="text-sm font-medium">DriveWealth Login</p>

      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" /> {error}
        </div>
      )}

      <div>
        <Label className="text-xs">Client ID</Label>
        <Input className="h-8 text-sm" value={clientId} onChange={(e) => setClientId(e.target.value)} placeholder="Client ID" />
      </div>

      <div>
        <Label className="text-xs">Client Secret</Label>
        <div className="relative">
          <Input className="h-8 text-sm pr-8" type={showPw ? "text" : "password"} value={clientSecret} onChange={(e) => setClientSecret(e.target.value)} placeholder="••••••••" />
          <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" onClick={() => setShowPw(!showPw)}>
            {showPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      <div>
        <Label className="text-xs">App Key</Label>
        <Input className="h-8 text-sm" value={appKey} onChange={(e) => setAppKey(e.target.value)} placeholder="App Key" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">User ID</Label>
          <Input className="h-8 text-sm" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="User ID" />
        </div>
        <div>
          <Label className="text-xs">Account ID</Label>
          <Input className="h-8 text-sm" value={accountId} onChange={(e) => setAccountId(e.target.value)} placeholder="Account ID" />
        </div>
      </div>

      <Button type="submit" className="w-full h-8 text-sm" disabled={isLoading || !clientId || !clientSecret || !appKey || !userId || !accountId}>
        {isLoading ? "Connecting…" : <><LogIn className="mr-1 h-3 w-3" /> Connect</>}
      </Button>
    </form>
  );
}
