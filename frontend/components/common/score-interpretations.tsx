"use client";

import { Z_SCORE_RANGES, M_SCORE_RANGES, F_SCORE_RANGES } from "@/lib/constants";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";

export function ScoreInterpretations() {
  return (
    <div className="grid grid-cols-1 gap-3 mb-4">
      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium w-full py-2 px-3 rounded-md bg-secondary hover:bg-accent">
          <ChevronDown className="h-4 w-4" />
          Altman Z-Score
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1 p-3 border rounded-md text-xs space-y-1">
          {Z_SCORE_RANGES.map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: r.color }} />
              <span className="font-medium">{r.label}</span>
              <span className="text-muted-foreground">— {r.description}</span>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium w-full py-2 px-3 rounded-md bg-secondary hover:bg-accent">
          <ChevronDown className="h-4 w-4" />
          Beneish M-Score
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1 p-3 border rounded-md text-xs space-y-1">
          {M_SCORE_RANGES.map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: r.color }} />
              <span className="font-medium">{r.label}</span>
              <span className="text-muted-foreground">— {r.description}</span>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium w-full py-2 px-3 rounded-md bg-secondary hover:bg-accent">
          <ChevronDown className="h-4 w-4" />
          Piotroski F-Score
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1 p-3 border rounded-md text-xs space-y-1">
          {F_SCORE_RANGES.map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: r.color }} />
              <span className="font-medium">{r.label}</span>
              <span className="text-muted-foreground">— {r.description}</span>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
