"use client";

import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { FileText, Database, Globe } from "lucide-react";
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
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-xs font-semibold">Knowledge Base Sources</Label>
        <div className="flex gap-2 text-xs">
          <button className="text-primary hover:underline" onClick={onSelectAll}>All</button>
          <button className="text-muted-foreground hover:underline" onClick={onDeselectAll}>None</button>
        </div>
      </div>

      <div className="max-h-48 overflow-y-auto space-y-1 rounded-md border p-2 bg-muted/20">
        {sources.length === 0 && (
          <p className="text-xs text-muted-foreground py-2 text-center">No knowledge base sources available</p>
        )}
        {sources.map((src) => (
          <div key={src.id} className="flex items-center gap-2 py-1 px-1 rounded hover:bg-muted/40">
            <Checkbox
              checked={selected.includes(src.id)}
              onCheckedChange={(v) => onToggle(src.id, v === true)}
            />
            <span className="text-muted-foreground">{SOURCE_ICONS[src.type] ?? <FileText className="h-3.5 w-3.5" />}</span>
            <span className="text-sm flex-1 truncate">{src.name}</span>
            <Badge variant="secondary" className="text-[10px] px-1.5">
              {src.chunk_count ?? 0} chunks
            </Badge>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-muted-foreground">
        {selected.length} of {sources.length} sources selected
      </p>
    </div>
  );
}
