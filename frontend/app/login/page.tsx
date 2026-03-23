"use client";

import { useState, useEffect } from "react";
import { useAuthStore } from "@/hooks/use-auth";
import { useRouter } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { AlertCircle, LogIn } from "lucide-react";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login, isAuthenticated, checkAuth } = useAuthStore();
  const router = useRouter();

  // If already authenticated, redirect (replace so /login leaves history)
  useEffect(() => {
    checkAuth().then(() => {
      if (useAuthStore.getState().isAuthenticated) {
        router.replace("/ind-stocks/fly-kite");
      }
    });
  }, [checkAuth, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(username, password);
      router.replace("/ind-stocks/fly-kite");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center app-background">
      <div className="w-full max-w-sm mx-auto">
        <div className="content-panel p-8 space-y-6">
          {/* Header */}
          <div className="text-center space-y-1">
            <h1 className="text-2xl font-bold tracking-tight">Centurion Capital</h1>
            <p className="text-sm text-muted-foreground">Sign in to your account</p>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 rounded-md p-3">
              <AlertCircle className="h-4 w-4 shrink-0" /> {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                autoComplete="username"
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                autoComplete="current-password"
              />
            </div>

            <Button type="submit" className="w-full" disabled={isSubmitting || !username || !password}>
              {isSubmitting ? "Signing in…" : <><LogIn className="mr-2 h-4 w-4" /> Sign In</>}
            </Button>
          </form>

          <p className="text-xs text-center text-muted-foreground">
            Centurion Capital LLC — Enterprise Trading Platform
          </p>
        </div>
      </div>
    </div>
  );
}
