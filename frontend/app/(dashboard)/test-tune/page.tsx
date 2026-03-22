"use client";

import { useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/common/spinner";
import { useTtsChapters, useTtsRun } from "@/hooks/use-tts";
import { TTS_CATEGORIES } from "@/lib/constants";
import { Play, Wrench } from "lucide-react";

export default function TestTunePage() {
  const chaptersQ = useTtsChapters();
  const { run, isRunning, progress, error } = useTtsRun();
  const [selected, setSelected] = useState<string[]>([]);

  const chapters = chaptersQ.data ?? [];
  const grouped = TTS_CATEGORIES.map((cat) => ({
    category: cat,
    items: chapters.filter((c) => c.category === cat),
  })).filter((g) => g.items.length > 0);

  const toggleChapter = (key: string, checked: boolean) => {
    setSelected((prev) => (checked ? [...prev, key] : prev.filter((k) => k !== key)));
  };

  const handleRun = () => {
    if (selected.length > 0) run(selected);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Wrench className="h-5 w-5 text-orange-500" />
        <h2 className="text-lg font-semibold">Test & Tune Trading Systems</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="content-panel p-4 space-y-4">
          <h3 className="text-sm font-semibold">Select Chapters</h3>
          {chaptersQ.isLoading ? (
            <Spinner />
          ) : (
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {grouped.map((g) => (
                <div key={g.category}>
                  <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">{g.category}</p>
                  {g.items.map((ch) => (
                    <div key={ch.key} className="flex items-center gap-2 py-0.5">
                      <Checkbox checked={selected.includes(ch.key)} onCheckedChange={(v) => toggleChapter(ch.key, v === true)} />
                      <Label className="text-sm font-normal">{ch.title}</Label>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={() => setSelected(chapters.map((c) => c.key))}>All</Button>
            <Button size="sm" variant="outline" onClick={() => setSelected([])}>None</Button>
          </div>

          <Button className="w-full" onClick={handleRun} disabled={isRunning || selected.length === 0}>
            {isRunning ? "Running…" : <><Play className="mr-1 h-4 w-4" /> Run ({selected.length})</>}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <div className="md:col-span-3 space-y-4">
          {isRunning && progress && (
            <div className="content-panel p-4">
              <Spinner label={`Running ${progress.completed}/${progress.total} chapters…`} />
            </div>
          )}

          {progress?.chapters &&
            Object.entries(progress.chapters).map(([key, ch]) => (
              <div key={key} className="content-panel p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold">{key}</h4>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    ch.status === "done" ? "bg-green-500/10 text-green-600" :
                    ch.status === "running" ? "bg-blue-500/10 text-blue-600" :
                    ch.status === "error" ? "bg-red-500/10 text-red-600" :
                    "bg-muted text-muted-foreground"
                  }`}>
                    {ch.status}
                  </span>
                </div>

                {ch.text_output && <pre className="text-xs bg-muted/50 rounded p-2 overflow-auto max-h-48">{ch.text_output}</pre>}
                {ch.error_message && <p className="text-sm text-destructive">{ch.error_message}</p>}

                {ch.figures.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {ch.figures.map((fig, i) => (
                      <img key={i} src={`data:image/png;base64,${fig}`} alt={`${key} figure ${i + 1}`} className="rounded border" />
                    ))}
                  </div>
                )}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
