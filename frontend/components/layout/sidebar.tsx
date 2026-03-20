"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { APP_MODULES, US_SUB_PAGES, IND_SUB_PAGES } from "@/lib/constants";
import {
  BarChart3, Globe, Brain, FlaskConical, Bitcoin, MessageSquare,
  ChevronDown, ChevronRight, Menu, X,
} from "lucide-react";
import { useState, useEffect } from "react";

const MODULE_ICONS: Record<string, React.ReactNode> = {
  "us-stocks": <BarChart3 className="h-4 w-4" />,
  "ind-stocks": <Globe className="h-4 w-4" />,
  "financial-ml": <Brain className="h-4 w-4" />,
  "test-tune": <FlaskConical className="h-4 w-4" />,
  "crypto": <Bitcoin className="h-4 w-4" />,
  "rag-engine": <MessageSquare className="h-4 w-4" />,
};

function getSubPages(key: string) {
  if (key === "us-stocks") return US_SUB_PAGES;
  if (key === "ind-stocks") return IND_SUB_PAGES;
  if (key === "financial-ml")
    return [
      { key: "main", label: "Chapters", href: "/financial-ml" },
      { key: "history", label: "History", href: "/financial-ml/history" },
    ];
  if (key === "test-tune")
    return [
      { key: "main", label: "Chapters", href: "/test-tune" },
      { key: "history", label: "History", href: "/test-tune/history" },
    ];
  return [];
}

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const ALWAYS_EXPANDED = new Set(["us-stocks", "ind-stocks"]);
  const [expandedModules, setExpandedModules] = useState<Set<string>>(
    new Set(APP_MODULES.map((m) => m.key))
  );
  const [mobileOpen, setMobileOpen] = useState(false);

  // Sync collapsed state to a CSS custom property so layout can react
  const sidebarWidth = collapsed ? "w-16" : "w-56";

  // Expose collapsed state via data attribute using an effect
  useEffect(() => {
    document.documentElement.setAttribute("data-sidebar-collapsed", String(collapsed));
  }, [collapsed]);

  const toggleModule = (key: string) => {
    setExpandedModules((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const activeModule = APP_MODULES.find((m) => pathname.startsWith(m.href))?.key;

  const navContent = (
    <nav className="flex flex-col h-full">
      {/* Logo / Brand */}
      <div className="px-4 py-4 border-b border-gray-700/50">
        <h2 className={cn("font-bold text-white transition-all", collapsed ? "text-sm text-center" : "text-lg")}>
          {collapsed ? "CC" : "Centurion Capital"}
        </h2>
        {!collapsed && <p className="text-[0.65rem] text-gray-400 tracking-wider uppercase">Trading Platform</p>}
      </div>

      {/* Module Links */}
      <div className="flex-1 overflow-y-auto py-2">
        {APP_MODULES.map((mod) => {
          const subPages = getSubPages(mod.key);
          const isActive = activeModule === mod.key;
          const isExpanded = expandedModules.has(mod.key);

          return (
            <div key={mod.key} className="mb-0.5">
              <div
                className={cn(
                  "flex items-center gap-2 px-4 py-2 cursor-pointer transition-colors text-sm",
                  isActive ? "text-[#58a6ff] bg-white/5" : "text-gray-400 hover:text-white hover:bg-white/5"
                )}
                onClick={() => subPages.length && !ALWAYS_EXPANDED.has(mod.key) ? toggleModule(mod.key) : undefined}
              >
                {MODULE_ICONS[mod.key]}
                {!collapsed && (
                  <>
                    <Link href={mod.href} className="flex-1" onClick={() => setMobileOpen(false)}>
                      {mod.label}
                    </Link>
                    {subPages.length > 0 && !ALWAYS_EXPANDED.has(mod.key) && (
                      <button onClick={(e) => { e.stopPropagation(); toggleModule(mod.key); }}>
                        {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                      </button>
                    )}
                  </>
                )}
              </div>

              {/* Sub-pages */}
              {!collapsed && (isExpanded || ALWAYS_EXPANDED.has(mod.key)) && subPages.length > 0 && (
                <div className="ml-6 border-l border-gray-700/50 pl-2">
                  {subPages.map((sp) => (
                    <Link
                      key={sp.key}
                      href={sp.href}
                      onClick={() => setMobileOpen(false)}
                      className={cn(
                        "block px-3 py-1.5 text-xs rounded-md transition-colors",
                        pathname === sp.href
                          ? "text-[#58a6ff] bg-[#58a6ff]/10 font-medium"
                          : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
                      )}
                    >
                      {sp.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Collapse toggle */}
      <div className="border-t border-gray-700/50 p-2 hidden lg:block">
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="w-full text-xs text-gray-500 hover:text-gray-300 py-1"
        >
          {collapsed ? "→" : "← Collapse"}
        </button>
      </div>
    </nav>
  );

  return (
    <>
      {/* Mobile toggle */}
      <button
        className="lg:hidden fixed top-3 left-3 z-50 p-2 rounded-md bg-gray-800 text-white shadow-lg"
        onClick={() => setMobileOpen((o) => !o)}
        aria-label="Toggle sidebar"
      >
        {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setMobileOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "sidebar fixed top-0 left-0 h-full z-40 transition-all duration-300",
          collapsed ? "w-16" : "w-56",
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {navContent}
      </aside>
    </>
  );
}
