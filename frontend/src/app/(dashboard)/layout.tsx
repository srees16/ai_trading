"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const authenticated = useAuthStore((s) => s.authenticated);
  const router = useRouter();

  useEffect(() => {
    if (!authenticated) {
      router.replace("/login");
    }
  }, [authenticated, router]);

  if (!authenticated) return null;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex flex-1 flex-col pl-60">
        <Header />
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
