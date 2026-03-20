"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, FileText, RefreshCw } from "lucide-react";
import type { RAGSource } from "@/lib/types";

interface KnowledgeBaseProps {
  sources: RAGSource[];
  isLoading: boolean;
  onRefresh: () => void;
  onDelete: (sourceId: string) => void;
}

export function KnowledgeBase({ sources, isLoading, onRefresh, onDelete }: KnowledgeBaseProps) {
  const totalChunks = sources.reduce((a, s) => a + (s.chunk_count ?? 0), 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Knowledge Base</p>
          <p className="text-xs text-muted-foreground">
            {sources.length} document{sources.length !== 1 ? "s" : ""} • {totalChunks} chunks
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh} disabled={isLoading}>
          <RefreshCw className={`h-3.5 w-3.5 mr-1 ${isLoading ? "animate-spin" : ""}`} /> Refresh
        </Button>
      </div>

      {sources.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center">
          <FileText className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">No documents ingested yet</p>
          <p className="text-xs text-muted-foreground">Upload PDFs or text files to build your knowledge base</p>
        </div>
      ) : (
        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          {sources.map((src) => (
            <div
              key={src.id}
              className="flex items-center gap-2 rounded border px-3 py-2 bg-card hover:bg-muted/30 transition-colors"
            >
              <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{src.name}</p>
                <p className="text-[11px] text-muted-foreground">
                  {src.type} • {src.chunk_count ?? 0} chunks
                  {src.created_at ? ` • ${new Date(src.created_at).toLocaleDateString()}` : ""}
                </p>
              </div>
              <Badge variant="secondary" className="text-[10px] shrink-0">
                {src.type}
              </Badge>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                onClick={() => onDelete(src.id)}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
