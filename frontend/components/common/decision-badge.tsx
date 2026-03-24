"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { DECISION_COLORS } from "@/lib/constants";
import type { DecisionTag } from "@/lib/types";

interface DecisionBadgeProps {
  decision: DecisionTag;
  className?: string;
}

export function DecisionBadge({ decision, className }: DecisionBadgeProps) {
  const color = DECISION_COLORS[decision] || "#999";
  const isLight = decision === "BUY" || decision === "HOLD";

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold",
        className
      )}
      style={{
        backgroundColor: color,
        color: isLight ? "#1a1a2e" : "#ffffff",
      }}
    >
      {decision.replace("_", " ")}
    </span>
  );
}
