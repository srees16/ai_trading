"use client";

import { useCallback, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X, FileText, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

interface PdfUploaderProps {
  onUpload: (files: File[]) => Promise<unknown>;
  acceptedTypes?: string[];
  maxSizeMB?: number;
}

interface FileStatus {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

export function PdfUploader({
  onUpload,
  acceptedTypes = [".pdf", ".txt", ".md", ".csv"],
  maxSizeMB = 25,
}: PdfUploaderProps) {
  const [files, setFiles] = useState<FileStatus[]>([]);
  const [isDrag, setIsDrag] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback(
    (file: File): string | null => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!acceptedTypes.includes(ext)) return `Unsupported type: ${ext}`;
      if (file.size > maxSizeMB * 1024 * 1024) return `File exceeds ${maxSizeMB}MB limit`;
      return null;
    },
    [acceptedTypes, maxSizeMB],
  );

  const addFiles = useCallback(
    (incoming: FileList | File[]) => {
      const arr = Array.from(incoming);
      const newFiles: FileStatus[] = arr.map((f) => {
        const err = validateFile(f);
        return { file: f, status: err ? "error" : "pending", error: err ?? undefined } as FileStatus;
      });
      setFiles((prev) => [...prev, ...newFiles]);
    },
    [validateFile],
  );

  const removeFile = (idx: number) => setFiles((prev) => prev.filter((_, i) => i !== idx));

  const handleUpload = async () => {
    const valid = files.filter((f) => f.status === "pending").map((f) => f.file);
    if (valid.length === 0) return;

    setUploading(true);
    setFiles((prev) => prev.map((f) => (f.status === "pending" ? { ...f, status: "uploading" } : f)));

    try {
      await onUpload(valid);
      setFiles((prev) => prev.map((f) => (f.status === "uploading" ? { ...f, status: "done" } : f)));
    } catch {
      setFiles((prev) =>
        prev.map((f) => (f.status === "uploading" ? { ...f, status: "error", error: "Upload failed" } : f)),
      );
    } finally {
      setUploading(false);
    }
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${isDrag ? "border-primary bg-primary/5" : "border-muted-foreground/30 hover:border-muted-foreground/50"}`}
        onDragOver={(e) => { e.preventDefault(); setIsDrag(true); }}
        onDragLeave={() => setIsDrag(false)}
        onDrop={(e) => { e.preventDefault(); setIsDrag(false); addFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
      >
        <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-sm">Drag & drop files or click to browse</p>
        <p className="text-xs text-muted-foreground mt-1">
          {acceptedTypes.join(", ")} — max {maxSizeMB}MB each
        </p>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple
          accept={acceptedTypes.join(",")}
          onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {files.map((fs, i) => (
            <div key={i} className="flex items-center gap-2 text-sm rounded border px-2 py-1.5 bg-muted/20">
              <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <span className="flex-1 truncate">{fs.file.name}</span>
              <span className="text-xs text-muted-foreground">{(fs.file.size / 1024).toFixed(0)} KB</span>
              {fs.status === "done" && <CheckCircle className="h-3.5 w-3.5 text-green-500" />}
              {fs.status === "error" && (
                <span className="text-xs text-destructive flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" /> {fs.error}
                </span>
              )}
              {fs.status === "uploading" && (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              )}
              {(fs.status === "pending" || fs.status === "error") && (
                <button className="text-muted-foreground hover:text-foreground" onClick={() => removeFile(i)}>
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Upload button */}
      {pendingCount > 0 && (
        <Button size="sm" className="w-full" onClick={handleUpload} disabled={uploading}>
          {uploading ? "Uploading…" : `Upload ${pendingCount} file(s)`}
        </Button>
      )}
    </div>
  );
}
