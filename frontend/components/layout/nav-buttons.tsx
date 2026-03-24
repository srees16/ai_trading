"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

interface NavButton {
  key: string;
  label: string;
  href: string;
}

interface NavButtonsProps {
  items: NavButton[];
  basePath?: string;
}

export function NavButtons({ items }: NavButtonsProps) {
  const pathname = usePathname();

  return (
    <div className="flex items-center gap-1 flex-wrap mb-2">
      {items.map((page) => {
        const isActive = pathname === page.href;
        return (
          <Link
            key={page.key}
            href={page.href}
            className={cn(
              "px-3 py-1.5 text-sm rounded-md transition-colors font-medium",
              isActive
                ? "bg-primary text-primary-foreground shadow-sm"
                : "bg-secondary text-secondary-foreground hover:bg-accent"
            )}
          >
            {page.label}
          </Link>
        );
      })}
    </div>
  );
}
