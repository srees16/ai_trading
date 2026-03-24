"use client";

import { Spinner } from "@/components/common/spinner";
import { Badge } from "@/components/ui/badge";
import { useTtsHistory } from "@/hooks/use-tts";
import { Wrench } from "lucide-react";

export default function TTSHistoryPage() {
  const historyQ = useTtsHistory();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Wrench className="h-5 w-5 text-orange-500" />
        <h2 className="text-lg font-semibold">Test & Tune — Run History</h2>
      </div>

      {historyQ.isLoading && <Spinner />}

      {historyQ.data && (
        <div className="content-panel p-4">
          <p className="text-xs text-muted-foreground mb-3">
            {historyQ.data.total} total runs
          </p>

          {historyQ.data.data.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No batch runs yet. Go to the Chapters tab to run experiments.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground text-left">
                    <th className="py-2 pr-3">Date</th>
                    <th className="py-2 pr-3">Batch ID</th>
                    <th className="py-2 pr-3">Status</th>
                    <th className="py-2 pr-3">Chapters</th>
                    <th className="py-2 pr-3">Progress</th>
                  </tr>
                </thead>
                <tbody>
                  {historyQ.data.data.map((row) => (
                    <tr key={row.batch_id} className="border-b">
                      <td className="py-2 pr-3 text-xs">
                        {new Date(row.created_at).toLocaleString()}
                      </td>
                      <td className="py-2 pr-3 font-mono text-xs">
                        {row.batch_id.slice(0, 8)}…
                      </td>
                      <td className="py-2 pr-3">
                        <Badge
                          variant={
                            row.status === "completed"
                              ? "default"
                              : row.status === "error"
                              ? "destructive"
                              : "secondary"
                          }
                        >
                          {row.status}
                        </Badge>
                      </td>
                      <td className="py-2 pr-3 text-xs">
                        {row.chapters.join(", ")}
                      </td>
                      <td className="py-2 pr-3">
                        {row.completed}/{row.total}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {historyQ.error && (
        <p className="text-sm text-destructive">{historyQ.error.message}</p>
      )}
    </div>
  );
}
