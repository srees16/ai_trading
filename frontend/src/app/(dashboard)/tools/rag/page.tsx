"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ragApi, type RetrievedChunk } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MetricCard } from "@/components/shared/metric-card";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Upload, Search, Trash2, FileText, Database, MessageSquare, Settings2, RefreshCw } from "lucide-react";
import { toast } from "sonner";

export default function RAGPage() {
  // Stats
  const [stats, setStats] = useState<{ total_chunks: number; sources: string[]; embedding_model: string; collection_name: string } | null>(null);
  const [config, setConfig] = useState<{ embedding_model: string; llm_provider: string; chunk_size: number; chunk_overlap: number; top_k: number } | null>(null);
  // Query
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [sourceFilter, setSourceFilter] = useState("");
  const [querying, setQuerying] = useState(false);
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<RetrievedChunk[]>([]);
  // Upload
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Sources
  const [sources, setSources] = useState<string[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refreshStats = useCallback(async () => {
    try {
      const [statsRes, cfgRes, srcRes] = await Promise.all([ragApi.stats(), ragApi.config(), ragApi.sources()]);
      if (statsRes.success) setStats({ total_chunks: statsRes.total_chunks, sources: statsRes.sources, embedding_model: statsRes.embedding_model, collection_name: statsRes.collection_name });
      if (cfgRes.success) setConfig({ embedding_model: cfgRes.embedding_model, llm_provider: cfgRes.llm_provider, chunk_size: cfgRes.chunk_size, chunk_overlap: cfgRes.chunk_overlap, top_k: cfgRes.top_k });
      if (srcRes.success) setSources(srcRes.data);
    } catch {}
  }, []);

  useEffect(() => { refreshStats(); }, [refreshStats]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await ragApi.ingest(file);
      if (res.success) {
        toast.success(`Ingested "${res.filename}" — ${res.chunks_created} chunks`);
        refreshStats();
      }
    } catch (err: any) { toast.error(err?.message || "Upload failed"); }
    finally { setUploading(false); if (fileInputRef.current) fileInputRef.current.value = ""; }
  };

  const handleQuery = async () => {
    if (!query.trim()) { toast.error("Enter a query"); return; }
    setQuerying(true);
    setAnswer("");
    setChunks([]);
    try {
      const res = await ragApi.query(query, topK, sourceFilter || undefined);
      if (res.success) {
        setAnswer(res.answer);
        setChunks(res.chunks);
      }
    } catch (err: any) { toast.error(err?.message || "Query failed"); }
    finally { setQuerying(false); }
  };

  const handleDeleteSource = async (source: string) => {
    setDeleting(source);
    try {
      const res = await ragApi.deleteDocument(source);
      if (res.success) { toast.success(`Deleted: ${source}`); refreshStats(); }
    } catch (err: any) { toast.error(err?.message || "Delete failed"); }
    finally { setDeleting(null); }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">RAG Knowledge Engine</h1>
        <p className="text-sm text-muted-foreground">Upload documents, query with AI, and manage your knowledge base</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard title="Total Chunks" value={String(stats?.total_chunks || 0)} icon={<Database className="h-4 w-4" />} />
        <MetricCard title="Documents" value={String(sources.length)} icon={<FileText className="h-4 w-4" />} />
        <MetricCard title="Embedding Model" value={stats?.embedding_model || "—"} />
        <MetricCard title="LLM Provider" value={config?.llm_provider || "—"} />
      </div>

      <Tabs defaultValue="query">
        <TabsList>
          <TabsTrigger value="query"><MessageSquare className="mr-1 h-3 w-3" />Query</TabsTrigger>
          <TabsTrigger value="upload"><Upload className="mr-1 h-3 w-3" />Upload</TabsTrigger>
          <TabsTrigger value="sources"><FileText className="mr-1 h-3 w-3" />Sources ({sources.length})</TabsTrigger>
          <TabsTrigger value="config"><Settings2 className="mr-1 h-3 w-3" />Config</TabsTrigger>
        </TabsList>

        {/* Query Tab */}
        <TabsContent value="query" className="space-y-4">
          <Card>
            <CardContent className="pt-6 space-y-4">
              <div>
                <Label className="text-xs">Ask a question</Label>
                <Textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} placeholder="e.g., What are the key risk management strategies?" />
              </div>
              <div className="flex items-end gap-4">
                <div className="w-24">
                  <Label className="text-xs">Top K</Label>
                  <Input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={1} max={20} />
                </div>
                <div className="flex-1">
                  <Label className="text-xs">Source Filter (optional)</Label>
                  <Select value={sourceFilter || "__all__"} onValueChange={(v) => setSourceFilter(v === "__all__" ? "" : v)}>
                    <SelectTrigger><SelectValue placeholder="All sources" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__all__">All sources</SelectItem>
                      {sources.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={handleQuery} disabled={querying}>
                  {querying ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                  Query
                </Button>
              </div>
            </CardContent>
          </Card>

          {answer && (
            <Card>
              <CardHeader><CardTitle className="text-base">Answer</CardTitle></CardHeader>
              <CardContent>
                <div className="prose prose-invert prose-sm max-w-none">
                  <p className="whitespace-pre-wrap">{answer}</p>
                </div>
                {chunks.length > 0 && (
                  <>
                    <Separator className="my-4" />
                    <p className="text-xs text-muted-foreground mb-2">Sources ({chunks.length} chunks retrieved):</p>
                    <ScrollArea className="h-64">
                      <div className="space-y-3">
                        {chunks.map((chunk, i) => (
                          <div key={i} className="rounded border p-3">
                            <div className="flex items-center justify-between mb-1">
                              <Badge variant="outline" className="text-xs">{chunk.source}</Badge>
                              <span className="text-xs text-muted-foreground">Chunk #{chunk.chunk_index} · Distance: {chunk.distance.toFixed(4)}</span>
                            </div>
                            <p className="text-xs text-muted-foreground whitespace-pre-wrap">{chunk.text}</p>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Upload Tab */}
        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Upload Document</CardTitle>
              <CardDescription>Upload PDF, TXT, MD, or DOCX files to your knowledge base</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 cursor-pointer hover:border-primary/50 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                {uploading ? (
                  <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
                ) : (
                  <Upload className="h-10 w-10 text-muted-foreground" />
                )}
                <p className="mt-2 text-sm text-muted-foreground">{uploading ? "Uploading & ingesting..." : "Click to upload or drag & drop"}</p>
                <p className="text-xs text-muted-foreground">PDF, TXT, MD, DOCX</p>
              </div>
              <input ref={fileInputRef} type="file" className="hidden" accept=".pdf,.txt,.md,.docx" onChange={handleUpload} />
              <Button variant="outline" onClick={async () => {
                try {
                  const res = await ragApi.reingest();
                  if (res.success) { toast.success(`Re-ingested ${res.total_chunks} chunks`); refreshStats(); }
                } catch (err: any) { toast.error(err?.message || "Re-ingest failed"); }
              }}>
                <RefreshCw className="mr-2 h-4 w-4" />Re-ingest All
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Sources Tab */}
        <TabsContent value="sources">
          <Card>
            <CardHeader><CardTitle className="text-base">Managed Sources</CardTitle></CardHeader>
            <CardContent>
              {sources.length > 0 ? (
                <div className="space-y-2">
                  {sources.map((src) => (
                    <div key={src} className="flex items-center justify-between rounded border p-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-mono">{src}</span>
                      </div>
                      <Button size="sm" variant="ghost" onClick={() => handleDeleteSource(src)} disabled={deleting === src}>
                        {deleting === src ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3 text-loss" />}
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No documents uploaded yet</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Config Tab */}
        <TabsContent value="config">
          <Card>
            <CardHeader><CardTitle className="text-base">RAG Configuration</CardTitle></CardHeader>
            <CardContent>
              {config ? (
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div><span className="text-muted-foreground">Embedding Model:</span> <span className="font-mono">{config.embedding_model}</span></div>
                  <div><span className="text-muted-foreground">LLM Provider:</span> <span className="font-bold">{config.llm_provider}</span></div>
                  <div><span className="text-muted-foreground">Chunk Size:</span> {config.chunk_size}</div>
                  <div><span className="text-muted-foreground">Chunk Overlap:</span> {config.chunk_overlap}</div>
                  <div><span className="text-muted-foreground">Top K:</span> {config.top_k}</div>
                  <div><span className="text-muted-foreground">Collection:</span> <span className="font-mono">{stats?.collection_name || "—"}</span></div>
                </div>
              ) : <p className="text-sm text-muted-foreground">Loading config...</p>}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
