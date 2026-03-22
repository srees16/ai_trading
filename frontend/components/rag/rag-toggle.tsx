"use client";

import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Brain, Search } from "lucide-react";

interface RagToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function RagToggle({ enabled, onToggle }: RagToggleProps) {
  return (
    <div className="flex items-center gap-3 rounded-md border p-2 bg-muted/40">
      <Search className="h-4 w-4 text-muted-foreground" />
      <div className="flex-1">
        <Label className="text-sm font-medium cursor-pointer" htmlFor="rag-toggle">
          RAG Mode
        </Label>
        <p className="text-xs text-muted-foreground">
          {enabled
            ? "Enable to use uploaded document(s) as context"
            : "Disable to use LLM without document(s) context"}
        </p>
      </div>
      <Switch id="rag-toggle" checked={enabled} onCheckedChange={onToggle} />
      <Brain className={`h-4 w-4 ${enabled ? "text-primary" : "text-muted-foreground"}`} />
    </div>
  );
}
