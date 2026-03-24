"use client";

import { useState } from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import type { ScreenerConfig } from "@/lib/types";

interface ScreenerConfigFormProps {
  onConfigChange: (config: ScreenerConfig) => void;
}

const SL_OPTIONS = [
  { value: "tighter", label: "Tighter (ATR-based)" },
  { value: "ma50", label: "50-day MA" },
  { value: "swing_low", label: "Swing Low" },
];

export function ScreenerConfigForm({ onConfigChange }: ScreenerConfigFormProps) {
  const [config, setConfig] = useState<ScreenerConfig>({
    min_price: 10,
    min_avg_volume: 500000,
    min_beta: 0.5,
    workers: 4,
    volume_multiplier: 1.5,
    lookback_days: 30,
    index_mode: false,
  });

  const update = (partial: Partial<ScreenerConfig>) => {
    const next = { ...config, ...partial };
    setConfig(next);
    onConfigChange(next);
  };

  return (
    <div className="space-y-4">
      {/* Price & Volume */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">Min Price ($)</Label>
          <Input type="number" className="h-8 text-sm" value={config.min_price} min={0} step={1} onChange={(e) => update({ min_price: Number(e.target.value) })} />
        </div>
        <div>
          <Label className="text-xs">Min Avg Volume</Label>
          <Input type="number" className="h-8 text-sm" value={config.min_avg_volume} min={0} step={10000} onChange={(e) => update({ min_avg_volume: Number(e.target.value) })} />
        </div>
      </div>

      {/* Beta & Volume Multiplier */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">Min Beta</Label>
          <Input type="number" className="h-8 text-sm" value={config.min_beta} min={0} max={5} step={0.1} onChange={(e) => update({ min_beta: Number(e.target.value) })} />
        </div>
        <div>
          <Label className="text-xs">Volume Multiplier</Label>
          <Input type="number" className="h-8 text-sm" value={config.volume_multiplier} min={1} max={10} step={0.1} onChange={(e) => update({ volume_multiplier: Number(e.target.value) })} />
        </div>
      </div>

      {/* Lookback & Workers */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">Lookback Days</Label>
          <Input type="number" className="h-8 text-sm" value={config.lookback_days} min={5} max={365} onChange={(e) => update({ lookback_days: Number(e.target.value) })} />
        </div>
        <div>
          <Label className="text-xs">Workers</Label>
          <Input type="number" className="h-8 text-sm" value={config.workers} min={1} max={16} onChange={(e) => update({ workers: Number(e.target.value) })} />
        </div>
      </div>

      {/* Index Mode */}
      <div className="flex items-center gap-2">
        <Checkbox checked={config.index_mode} onCheckedChange={(v) => update({ index_mode: v === true })} />
        <Label className="text-sm font-normal">Index Mode (use pre-defined index constituents)</Label>
      </div>
    </div>
  );
}
