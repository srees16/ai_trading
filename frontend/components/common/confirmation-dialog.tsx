"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle } from "lucide-react";
import { useState } from "react";

interface ConfirmationDialogProps {
  open?: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel?: () => void;
  variant?: "default" | "destructive";
  disabled?: boolean;
}

export function ConfirmationDialog({
  open = true,
  title,
  description,
  confirmLabel,
  onConfirm,
  onCancel,
  variant = "default",
  disabled = false,
}: ConfirmationDialogProps) {
  const [confirmed, setConfirmed] = useState(false);

  if (!open) return null;

  return (
    <Card className="border-amber-200 bg-amber-50/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <Checkbox
            id="confirm-action"
            checked={confirmed}
            onCheckedChange={(v) => setConfirmed(v === true)}
          />
          <label htmlFor="confirm-action" className="text-sm cursor-pointer">
            {description}
          </label>
        </div>
        <div className="flex gap-2">
          <Button
            variant={variant === "destructive" ? "destructive" : "default"}
            size="sm"
            disabled={!confirmed || disabled}
            onClick={() => { onConfirm(); setConfirmed(false); }}
          >
            {confirmLabel}
          </Button>
          {onCancel && (
            <Button variant="outline" size="sm" onClick={() => { onCancel(); setConfirmed(false); }}>
              Cancel
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
