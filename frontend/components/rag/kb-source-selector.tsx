"use client";

import { useState, useRef, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { FileText, Database, Globe, ChevronDown } from "lucide-react";
import type { RAGSource } from "@/lib/types";

interface KbSourceSelectorProps {
  sources: RAGSource[];
  selected: string[];
  onToggle: (sourceId: string, checked: boolean) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

const SOURCE_ICONS: Record<string, React.ReactNode> = {
  pdf: <FileText className="h-3.5 w-3.5" />,
  database: <Database className="h-3.5 w-3.5" />,
  web: <Globe className="h-3.5 w-3.5" />,
};

export function KbSourceSelector({ sources, selected, onToggle, onSelectAll, onDeselectAll }: KbSourceSelectorProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const label =
    selected.length === 0
      ? "All sources"
      : selected.length === sources.length
        ? "All sources"
        : `${selected.length} of ${sources.length} selected`;

  return (
    <div className="relative" ref={ref}>
      <p className="text-xs font-semibold mb-1">Context Source</p>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm bg-card hover:bg-muted/40 transition-colors"
      >
        <span className="truncate">{label}</span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover shadow-md">
          <div className="flex items-center justify-between px-3 py-2 border-b">
            <span className="text-xs text-muted-foreground">{sources.length} sources</span>
            <div className="flex gap-2 text-xs">
              <button className="text-primary hover:underline" onClick={onSelectAll}>All</button>
              <button className="text-muted-foreground hover:underline" onClick={onDeselectAll}>None</button>
            </div>
          </div>
          <div className="max-h-56 overflow-y-auto p-1">
            {sources.length === 0 && (
              <p className="text-xs text-muted-foreground py-3 text-center">No sources available</p>
            )}
            {sources.map((src) => (
              <label
                key={src.id}
                className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted/40 cursor-pointer"
              >
                <Checkbox
                  checked={selected.includes(src.id)}
                  onCheckedChange={(v) => onToggle(src.id, v === true)}
                />
                <span className="text-muted-foreground shrink-0">
                  {SOURCE_ICONS[src.type] ?? <FileText className="h-3.5 w-3.5" />}
                </span>
                <span className="text-sm flex-1 truncate">{src.name}</span>
                <Badge variant="secondary" className="text-[10px] px-1.5 shrink-0">
                  {src.chunk_count ?? 0}
                </Badge>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
