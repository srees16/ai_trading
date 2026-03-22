"use client";

import { useState } from "react";
import { QueryInput } from "@/components/rag/query-input";
import { RagToggle } from "@/components/rag/rag-toggle";
import { KbSourceSelector } from "@/components/rag/kb-source-selector";
import { StreamingAnswer } from "@/components/rag/streaming-answer";
import { PdfUploader } from "@/components/rag/pdf-uploader";
import { KnowledgeBase } from "@/components/rag/knowledge-base";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { useRagSources, useRagQuery } from "@/hooks/use-rag";
import { MessageSquare, Loader2 } from "lucide-react";

export default function RagEnginePage() {
  const [ragEnabled, setRagEnabled] = useState(true);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [tab, setTab] = useState("query");

  const { sources, isLoading, refresh, deleteSource, upload, isUploading, ingestTasks, isIngesting } = useRagSources();
  const { query, answer, chunks, isStreaming } = useRagQuery();

  const handleQuery = (q: string) => {
    query(q, ragEnabled, selectedSources.length > 0 ? selectedSources : undefined);
  };

  const toggleSource = (id: string, checked: boolean) => {
    setSelectedSources((prev) => (checked ? [...prev, id] : prev.filter((s) => s !== id)));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <MessageSquare className="h-5 w-5 text-blue-500" />
        <h2 className="text-lg font-semibold">RAG Engine</h2>
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="query">Query</TabsTrigger>
          <TabsTrigger value="manage">Knowledge Base</TabsTrigger>
        </TabsList>

        <TabsContent value="query" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="content-panel p-4 space-y-4">
              <RagToggle enabled={ragEnabled} onToggle={setRagEnabled} />
              {ragEnabled && (
                <KbSourceSelector
                  sources={sources}
                  selected={selectedSources}
                  onToggle={toggleSource}
                  onSelectAll={() => setSelectedSources(sources.map((s) => s.id))}
                  onDeselectAll={() => setSelectedSources([])}
                />
              )}
            </div>
            <div className="md:col-span-3 space-y-4">
              <div className="content-panel p-4">
                <QueryInput onSubmit={handleQuery} isLoading={isStreaming} />
              </div>
              <StreamingAnswer answer={answer} chunks={chunks} isStreaming={isStreaming} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="manage" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div className="content-panel p-4">
                <PdfUploader onUpload={upload} />
              </div>
              {ingestTasks.length > 0 && (
                <div className="content-panel p-4 space-y-2">
                  <p className="text-sm font-medium flex items-center gap-2">
                    {isIngesting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    Ingestion Status
                  </p>
                  {ingestTasks.map((t) => (
                    <div key={t.task_id} className="rounded border px-3 py-2 bg-muted/20 space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="truncate font-medium">{t.file_name}</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                          t.status === "completed" ? "bg-green-500/10 text-green-600" :
                          t.status === "failed" ? "bg-red-500/10 text-red-600" :
                          "bg-blue-500/10 text-blue-600"
                        }`}>{t.status}</span>
                      </div>
                      {(t.status === "running" || t.status === "pending") && (
                        <>
                          <Progress value={Math.round(t.stage_pct * 100)} className="h-1.5" />
                          <p className="text-[11px] text-muted-foreground">{t.stage}</p>
                        </>
                      )}
                      {t.error && <p className="text-[11px] text-destructive">{t.error}</p>}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="content-panel p-4">
              <KnowledgeBase
                sources={sources}
                isLoading={isLoading}
                onRefresh={() => refresh()}
                onDelete={deleteSource}
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
