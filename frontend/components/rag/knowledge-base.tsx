"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, FileText, RefreshCw, Clock, Layers, FileType, HardDrive } from "lucide-react";
import type { RAGSource } from "@/lib/types";

interface KnowledgeBaseProps {
  sources: RAGSource[];
  isLoading: boolean;
  onRefresh: () => void;
  onDelete: (sourceId: string) => void;
}

function formatFileSize(bytes: number | null | undefined): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
  try {
    const n = Number(ts);
    const d = isNaN(n) ? new Date(ts) : new Date(n * 1000);
    return d.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return "—";
  }
}

export function KnowledgeBase({ sources, isLoading, onRefresh, onDelete }: KnowledgeBaseProps) {
  const totalChunks = sources.reduce((a, s) => a + (s.chunk_count ?? 0), 0);
  const totalPages = sources.reduce((a, s) => a + (s.page_count ?? 0), 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Ingested Documents</p>
          <p className="text-xs text-muted-foreground">
            {sources.length} source{sources.length !== 1 ? "s" : ""}
            {totalPages > 0 ? ` • ${totalPages} pages` : ""}
            {" "}• {totalChunks} chunks
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
        <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
          {sources.map((src) => (
            <div
              key={src.id}
              className="rounded-lg border px-3 py-2.5 bg-card hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                <p className="text-sm font-medium truncate flex-1">{src.name}</p>
                <Badge variant="secondary" className="text-[10px] shrink-0 uppercase">
                  {src.type}
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive shrink-0"
                  onClick={() => onDelete(src.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1.5 ml-6 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Layers className="h-3 w-3" /> {src.chunk_count ?? 0} chunks
                </span>
                {src.page_count != null && (
                  <span className="flex items-center gap-1">
                    <FileType className="h-3 w-3" /> {src.page_count} pages
                  </span>
                )}
                {src.file_size_bytes != null && (
                  <span className="flex items-center gap-1">
                    <HardDrive className="h-3 w-3" /> {formatFileSize(src.file_size_bytes)}
                  </span>
                )}
                {src.ingested_at && (
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" /> {formatTimestamp(src.ingested_at)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
