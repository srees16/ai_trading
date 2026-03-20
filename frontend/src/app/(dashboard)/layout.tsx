"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore, useAppStore } from "@/lib/store";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";

// Lazy-load market data widgets — not needed for initial page render
const TickerRibbon = dynamic(
  () => import("@/components/shared/ticker-ribbon").then((m) => ({ default: m.TickerRibbon })),
  { ssr: false }
);
const VIXBand = dynamic(
  () => import("@/components/shared/vix-band").then((m) => ({ default: m.VIXBand })),
  { ssr: false }
);

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const authenticated = useAuthStore((s) => s.authenticated);
  const hydrated = useAuthStore((s) => s._hydrated);
  const router = useRouter();
  const pathname = usePathname();
  const setMarket = useAppStore((s) => s.setMarket);
  const [ready, setReady] = useState(false);

  // Sync market toggle with current URL on load/navigation
  useEffect(() => {
    if (pathname.startsWith("/ind")) {
      setMarket("IND");
    } else if (pathname.startsWith("/us")) {
      setMarket("US");
    }
  }, [pathname, setMarket]);

  useEffect(() => {
    if (!hydrated) return;
    if (!authenticated) {
      router.replace("/login");
    } else {
      setReady(true);
    }
  }, [hydrated, authenticated, router]);

  // Show the shell skeleton instantly while hydrating (prevents blank flash)
  if (!ready) {
    return (
      <div className="flex min-h-screen">
        <div className="w-60 border-r border-border bg-card" />
        <div className="flex flex-1 flex-col pl-60">
          <div className="h-14 border-b border-border" />
          <main className="flex-1 overflow-auto p-6">
            <div className="mx-auto max-w-7xl" />
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex flex-1 flex-col pl-60">
        <Header />
        <TickerRibbon />
        <VIXBand />
        <main className="flex-1 overflow-auto p-6">
          <div className="mx-auto max-w-7xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
