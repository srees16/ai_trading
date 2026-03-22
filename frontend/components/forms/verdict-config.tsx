"use client";

import { useState } from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { VERDICT_LAYERS, SL_METHODS, OUTPUT_FORMATS } from "@/lib/constants";

interface VerdictConfigProps {
  onConfigChange: (config: {
    layers: string[];
    risk_pct: number;
    sl_method: string;
    output_format: string;
    custom_notes: string;
  }) => void;
}

export function VerdictConfig({ onConfigChange }: VerdictConfigProps) {
  const [layers, setLayers] = useState<string[]>([...VERDICT_LAYERS]);
  const [riskPct, setRiskPct] = useState(2);
  const [slMethod, setSlMethod] = useState("atr");
  const [outputFormat, setOutputFormat] = useState("detailed");
  const [notes, setNotes] = useState("");

  const emitChange = (overrides?: Record<string, unknown>) => {
    onConfigChange({ layers, risk_pct: riskPct, sl_method: slMethod, output_format: outputFormat, custom_notes: notes, ...overrides });
  };

  const toggleLayer = (value: string, checked: boolean) => {
    const next = checked ? [...layers, value] : layers.filter((l) => l !== value);
    setLayers(next);
    emitChange({ layers: next });
  };

  return (
    <div className="space-y-4">
      {/* Layers */}
      <div>
        <Label className="text-xs font-semibold">Analysis Layers</Label>
        <div className="grid grid-cols-2 gap-2 mt-1">
          {VERDICT_LAYERS.map((l) => (
            <div key={l} className="flex items-center gap-2">
              <Checkbox checked={layers.includes(l)} onCheckedChange={(v) => toggleLayer(l, v === true)} />
              <Label className="text-sm font-normal capitalize">{l.replace("_", " ")}</Label>
            </div>
          ))}
        </div>
      </div>

      {/* Risk % */}
      <div>
        <Label className="text-xs">Risk per Trade (%)</Label>
        <Input type="number" className="h-8 text-sm" value={riskPct} min={0.5} max={10} step={0.5} onChange={(e) => { setRiskPct(Number(e.target.value)); emitChange({ risk_pct: Number(e.target.value) }); }} />
      </div>

      {/* SL Method */}
      <div>
        <Label className="text-xs">Stop-Loss Method</Label>
        <Select value={slMethod} onValueChange={(v) => { setSlMethod(v); emitChange({ sl_method: v }); }}>
          <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            {SL_METHODS.map((m) => <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {/* Output Format */}
      <div>
        <Label className="text-xs">Output Format</Label>
        <Select value={outputFormat} onValueChange={(v) => { setOutputFormat(v); emitChange({ output_format: v }); }}>
          <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            {OUTPUT_FORMATS.map((f) => <SelectItem key={f.value} value={f.value}>{f.label}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {/* Custom Notes */}
      <div>
        <Label className="text-xs">Custom Notes</Label>
        <Input className="h-8 text-sm" value={notes} onChange={(e) => { setNotes(e.target.value); emitChange({ custom_notes: e.target.value }); }} placeholder="Optional notes for the verdict engine" />
      </div>
    </div>
  );
}
