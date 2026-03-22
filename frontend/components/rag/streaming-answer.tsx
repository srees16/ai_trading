"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ThumbsUp, ThumbsDown, Copy, Check, Loader2, Search } from "lucide-react";
import type { RAGChunk } from "@/lib/types";

interface StreamingAnswerProps {
  answer: string;
  chunks: RAGChunk[];
  isStreaming: boolean;
}

export function StreamingAnswer({ answer, chunks, isStreaming }: StreamingAnswerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [startTime] = useState(() => Date.now());
  const [ttft, setTtft] = useState<number | null>(null);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const [copied, setCopied] = useState(false);
  const ttftSet = useRef(false);

  useEffect(() => {
    if (containerRef.current && isStreaming) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [answer, isStreaming]);

  // Track TTFT (time to first token)
  useEffect(() => {
    if (answer.length > 0 && !ttftSet.current) {
      setTtft(Date.now() - startTime);
      ttftSet.current = true;
    }
  }, [answer, startTime]);

  // Reset on new query
  useEffect(() => {
    if (isStreaming && !answer) {
      ttftSet.current = false;
      setTtft(null);
      setFeedback(null);
      setCopied(false);
    }
  }, [isStreaming, answer]);

  const handleCopy = () => {
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!answer && !isStreaming) return null;

  return (
    <div className="space-y-3">
      {/* TTFT metric */}
      {ttft !== null && (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>TTFT: <strong>{(ttft / 1000).toFixed(2)}s</strong></span>
          {isStreaming && <span className="animate-pulse">Streaming…</span>}
          {!isStreaming && <span>{answer.split(/\s+/).length} words</span>}
        </div>
      )}

      {/* Answer */}
      <div
        ref={containerRef}
        className="rounded-lg border bg-card p-4 max-h-[420px] overflow-y-auto prose prose-sm dark:prose-invert"
      >
        {answer ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <div className="text-center space-y-1">
              {chunks.length > 0 ? (
                <p className="text-sm text-muted-foreground animate-pulse">Generating answer…</p>
              ) : (
                <>
                  <p className="text-sm font-medium flex items-center gap-1.5 justify-center">
                    <Search className="h-3.5 w-3.5" /> Retrieving context…
                  </p>
                  <p className="text-xs text-muted-foreground">Searching knowledge base for relevant chunks</p>
                </>
              )}
            </div>
          </div>
        )}
        {isStreaming && <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-0.5" />}
      </div>

      {/* Feedback buttons */}
      {answer && !isStreaming && (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={feedback === "up" ? "default" : "outline"}
            className="h-7 px-2"
            onClick={() => setFeedback("up")}
          >
            <ThumbsUp className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            variant={feedback === "down" ? "default" : "outline"}
            className="h-7 px-2"
            onClick={() => setFeedback("down")}
          >
            <ThumbsDown className="h-3.5 w-3.5" />
          </Button>
          <Button size="sm" variant="outline" className="h-7 px-2" onClick={handleCopy}>
            {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          </Button>
        </div>
      )}

      {/* Source Chunks (Citations) */}
      {chunks.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-muted-foreground">Sources ({chunks.length})</p>
          <div className="grid gap-2">
            {chunks.map((chunk, i) => (
              <div key={i} className="text-xs rounded border p-2 bg-muted/30">
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant="outline" className="text-[10px]">
                    [{i + 1}] {chunk.source}
                  </Badge>
                  {chunk.score !== undefined && (
                    <span className="text-[10px] text-muted-foreground">
                      relevance: {(chunk.score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <p className="text-muted-foreground line-clamp-3">{chunk.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
