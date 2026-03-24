"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DecisionBadge } from "@/components/common/decision-badge";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import { getTopSignals, formatNumber } from "@/lib/utils";
import type { TradingSignal } from "@/lib/types";

interface TopSignalsProps {
  signals: TradingSignal[];
}

function SignalCard({ signal }: { signal: TradingSignal }) {
  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full">
        <div className="flex items-center justify-between p-3 rounded-md border hover:bg-muted/30 transition-colors">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{signal.news_item.ticker}</span>
            <DecisionBadge decision={signal.decision} />
            <span className="text-sm text-muted-foreground">{formatNumber(signal.decision_score, 3)}</span>
          </div>
          <ChevronDown className="h-4 w-4" />
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-3 pb-3 text-sm space-y-1">
        <p><span className="font-medium">Sentiment:</span> {signal.news_item.sentiment_label} ({(signal.news_item.sentiment_confidence * 100).toFixed(1)}%)</p>
        <p><span className="font-medium">Reasoning:</span> {signal.reasoning}</p>
        <p className="text-xs text-muted-foreground truncate">{signal.news_item.title}</p>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function TopSignals({ signals }: TopSignalsProps) {
  const { topBuy, topSell } = getTopSignals(signals);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-trade-green">🔵 Top Buy Signals</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {topBuy.length === 0 ? (
            <p className="text-sm text-muted-foreground">No buy signals</p>
          ) : (
            topBuy.map((s, i) => <SignalCard key={i} signal={s} />)
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-trade-red">🔴 Top Sell Signals</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {topSell.length === 0 ? (
            <p className="text-sm text-muted-foreground">No sell signals</p>
          ) : (
            topSell.map((s, i) => <SignalCard key={i} signal={s} />)
          )}
        </CardContent>
      </Card>
    </div>
  );
}
