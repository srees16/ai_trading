"use client";

import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface RagToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function RagToggle({ enabled, onToggle }: RagToggleProps) {
  return (
    <div className="flex items-center gap-3 rounded-md border p-2 bg-muted/40">
      <div className="flex-1">
        <Label className="text-sm font-medium cursor-pointer" htmlFor="rag-toggle">
          RAG Mode
        </Label>
        <p className="text-xs text-muted-foreground">
          {enabled
            ? "Using uploaded document(s) as context"
            : "Using LLM without document context"}
        </p>
      </div>
      <Switch id="rag-toggle" checked={enabled} onCheckedChange={onToggle} />
    </div>
  );
}
