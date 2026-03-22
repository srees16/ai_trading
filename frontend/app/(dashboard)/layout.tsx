"use client";

import { useEffect, useRef } from "react";
import { useAuthStore } from "@/hooks/use-auth";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { HeaderBar } from "@/components/layout/header-bar";
import { Footer } from "@/components/layout/footer";
import { Spinner } from "@/components/common/spinner";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, checkAuth, heartbeat } = useAuthStore();
  const router = useRouter();
  const lastHeartbeat = useRef(0);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  useEffect(() => {
    const handleActivity = () => {
      const now = Date.now();
      if (now - lastHeartbeat.current > 60_000) {
        lastHeartbeat.current = now;
        heartbeat();
      }
    };
    window.addEventListener("mousemove", handleActivity);
    window.addEventListener("keydown", handleActivity);
    return () => {
      window.removeEventListener("mousemove", handleActivity);
      window.removeEventListener("keydown", handleActivity);
    };
  }, [heartbeat]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center app-background">
        <Spinner />
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen app-background">
      <Sidebar />
      <div className="sidebar-content-offset flex flex-col min-h-screen transition-all duration-300">
        <HeaderBar />
        <main className="flex-1 p-4 lg:p-6 overflow-x-hidden">
          {children}
        </main>
        <Footer />
      </div>
    </div>
  );
}
