"use client";

import { SPINNER_MESSAGES } from "@/lib/constants";
import { randomPick } from "@/lib/utils";
import { useState, useEffect } from "react";

interface SpinnerProps {
  label?: string;
  messages?: string[];
}

export function Spinner({ label, messages = SPINNER_MESSAGES }: SpinnerProps) {
  const [message, setMessage] = useState(label || messages[0]);

  useEffect(() => {
    if (!label) setMessage(randomPick(messages));
  }, [label, messages]);

  return (
    <div className="flex items-center gap-3 py-4">
      <div className="spinner-ring" />
      <span className="text-sm text-muted-foreground italic">{message}</span>
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-12 bg-muted rounded-lg" />
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-24 bg-muted rounded-lg" />
        ))}
      </div>
      <div className="h-64 bg-muted rounded-lg" />
    </div>
  );
}
