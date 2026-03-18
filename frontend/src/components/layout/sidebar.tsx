"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/lib/store";
import {
  BarChart3,
  LineChart,
  TrendingUp,
  History,
  Briefcase,
  Award,
  Settings,
  Bot,
  Bitcoin,
  Activity,
  Target,
  LayoutGrid,
  IndianRupee,
  DollarSign,
  Layers,
  FlaskConical,
  SlidersHorizontal,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

interface NavItem {
  title: string;
  href: string;
  icon: React.ElementType;
  market?: "US" | "IND" | "ALL";
}

const mainNav: NavItem[] = [
  { title: "Dashboard", href: "/dashboard", icon: LayoutGrid, market: "ALL" },
];

const usNav: NavItem[] = [
  { title: "Analysis", href: "/us/analysis", icon: BarChart3 },
  { title: "Fundamentals", href: "/us/fundamentals", icon: TrendingUp },
  { title: "Backtesting", href: "/us/backtesting", icon: LineChart },
  { title: "Verdict", href: "/us/verdict", icon: Award },
  { title: "Holdings", href: "/us/holdings", icon: Briefcase },
  { title: "History", href: "/us/history", icon: History },
];

const indNav: NavItem[] = [
  { title: "Analysis", href: "/ind/analysis", icon: BarChart3 },
  { title: "Screener", href: "/ind/screener", icon: Target },
  { title: "Options", href: "/ind/options", icon: Layers },
  { title: "Trading", href: "/ind/trading", icon: Activity },
  { title: "Backtesting", href: "/ind/backtesting", icon: LineChart },
  { title: "History", href: "/ind/history", icon: History },
];

const toolsNav: NavItem[] = [
  { title: "Financial ML", href: "/tools/finance-ml", icon: FlaskConical, market: "ALL" },
  { title: "Test & Tune", href: "/tools/test-tune", icon: SlidersHorizontal, market: "ALL" },
  { title: "Crypto", href: "/tools/crypto", icon: Bitcoin, market: "ALL" },
  { title: "RAG Engine", href: "/tools/rag", icon: Bot, market: "ALL" },
];

const bottomNav: NavItem[] = [
  { title: "Settings", href: "/settings", icon: Settings, market: "ALL" },
];

function NavSection({ title, items }: { title: string; items: NavItem[] }) {
  const pathname = usePathname();

  return (
    <div className="mb-4">
      <h4 className="mb-1 px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/60">
        {title}
      </h4>
      <nav className="space-y-0.5">
        {items.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              <span>{item.title}</span>
              {isActive && <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}

export function Sidebar() {
  const { market, setMarket } = useAppStore();

  return (
    <aside className="fixed left-0 top-0 z-30 flex h-screen w-60 flex-col border-r bg-card">
      {/* Logo */}
      <div className="flex h-14 items-center gap-2 border-b px-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20">
          <TrendingUp className="h-4 w-4 text-primary" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-bold tracking-tight">Centurion</span>
          <span className="text-[10px] text-muted-foreground">Capital LLC</span>
        </div>
      </div>

      {/* Market Toggle */}
      <div className="px-3 py-3">
        <div className="flex rounded-lg bg-muted p-0.5">
          <button
            onClick={() => setMarket("US")}
            className={cn(
              "flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-all",
              market === "US" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <DollarSign className="h-3.5 w-3.5" />
            US Stocks
          </button>
          <button
            onClick={() => setMarket("IND")}
            className={cn(
              "flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-all",
              market === "IND" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <IndianRupee className="h-3.5 w-3.5" />
            IND Stocks
          </button>
        </div>
      </div>

      <Separator />

      {/* Nav */}
      <ScrollArea className="flex-1 px-2 py-3">
        <NavSection title="Main" items={mainNav} />
        {market === "US" && <NavSection title="US Markets" items={usNav} />}
        {market === "IND" && <NavSection title="Indian Markets" items={indNav} />}
        <NavSection title="Tools" items={toolsNav} />
      </ScrollArea>

      {/* Bottom */}
      <Separator />
      <div className="px-2 py-2">
        <NavSection title="" items={bottomNav} />
      </div>
    </aside>
  );
}
