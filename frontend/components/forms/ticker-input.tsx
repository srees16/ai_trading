"use client";

import { useState, useCallback } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { ChevronDown, Upload, Download, CheckCircle, AlertTriangle } from "lucide-react";
import { parseTickers, parseTickerCSV, createSampleCSV, isValidTicker, downloadFile } from "@/lib/utils";

interface TickerInputProps {
  defaultTickers: string[];
  onTickersChange: (tickers: string[]) => void;
  manualDefault?: string;
  prefix?: string;
}

export function TickerInput({ defaultTickers, onTickersChange, manualDefault = "GOOGL, TSLA", prefix = "" }: TickerInputProps) {
  const [mode, setMode] = useState<string>("default");
  const [manualText, setManualText] = useState(manualDefault);
  const [uploadedTickers, setUploadedTickers] = useState<string[]>([]);
  const [invalidTickers, setInvalidTickers] = useState<string[]>([]);

  const handleModeChange = useCallback((value: string) => {
    setMode(value);
    if (value === "default") {
      onTickersChange(defaultTickers);
      setInvalidTickers([]);
    } else if (value === "manual") {
      const tickers = parseTickers(manualText);
      const valid = tickers.filter(isValidTicker);
      const invalid = tickers.filter((t) => !isValidTicker(t));
      setInvalidTickers(invalid);
      onTickersChange(valid);
    }
  }, [defaultTickers, manualText, onTickersChange]);

  const handleManualChange = useCallback((text: string) => {
    setManualText(text);
    const tickers = parseTickers(text);
    const valid = tickers.filter(isValidTicker);
    const invalid = tickers.filter((t) => !isValidTicker(t));
    setInvalidTickers(invalid);
    onTickersChange(valid);
  }, [onTickersChange]);

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const tickers = await parseTickerCSV(file);
    setUploadedTickers(tickers);
    onTickersChange(tickers);
  }, [onTickersChange]);

  return (
    <div className="space-y-3">
      <Label className="text-sm font-medium">Input Method</Label>
      <RadioGroup value={mode} onValueChange={handleModeChange} className="flex flex-wrap gap-x-4 gap-y-2">
        <div className="flex items-center gap-1.5">
          <RadioGroupItem value="default" id={`${prefix}default`} />
          <Label htmlFor={`${prefix}default`} className="text-sm cursor-pointer">Default Tickers</Label>
        </div>
        <div className="flex items-center gap-1.5">
          <RadioGroupItem value="manual" id={`${prefix}manual`} />
          <Label htmlFor={`${prefix}manual`} className="text-sm cursor-pointer">Manual Entry</Label>
        </div>
        <div className="flex items-center gap-1.5">
          <RadioGroupItem value="csv" id={`${prefix}csv`} />
          <Label htmlFor={`${prefix}csv`} className="text-sm cursor-pointer">Upload CSV</Label>
        </div>
      </RadioGroup>

      {mode === "default" && (
        <Collapsible>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            <ChevronDown className="h-3 w-3" />
            View default tickers ({defaultTickers.length})
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-1 flex flex-wrap gap-1">
            {defaultTickers.map((t) => (
              <Badge key={t} variant="secondary" className="text-xs">{t}</Badge>
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}

      {mode === "manual" && (
        <div className="space-y-2">
          <Textarea
            value={manualText}
            onChange={(e) => handleManualChange(e.target.value)}
            placeholder="Enter tickers separated by commas…"
            className="h-20"
          />
          {invalidTickers.length > 0 && (
            <div className="flex items-center gap-1 text-xs text-amber-600">
              <AlertTriangle className="h-3 w-3" />
              Invalid: {invalidTickers.join(", ")}
            </div>
          )}
        </div>
      )}

      {mode === "csv" && (
        <div className="space-y-2">
          <div className="border-2 border-dashed rounded-md p-4 text-center">
            <Upload className="h-6 w-6 mx-auto text-muted-foreground mb-2" />
            <label className="cursor-pointer text-sm text-primary hover:underline">
              Choose CSV file
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => downloadFile(createSampleCSV(), "sample_tickers.csv")}
          >
            <Download className="h-3.5 w-3.5 mr-1.5" />
            Download Sample CSV
          </Button>
          {uploadedTickers.length > 0 && (
            <div className="flex items-center gap-1 text-xs text-green-600">
              <CheckCircle className="h-3 w-3" />
              {uploadedTickers.length} tickers loaded
            </div>
          )}
        </div>
      )}
    </div>
  );
}
