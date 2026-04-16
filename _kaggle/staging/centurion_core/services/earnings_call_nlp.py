"""
Earnings Call Transcript NLP — T4-6.

Process earnings call transcripts through NLP pipeline:
1. Download/fetch transcript text (from APIs or RAG uploads)
2. Run FinBERT sentiment analysis on management discussion
3. Extract forward guidance signals (growth language, caution markers)
4. Generate pre-earnings positioning forecasts

Research basis:
  - Li (2010): "The Information Content of Forward-Looking Statements"
  - Jiang, Lee, Martin & Zhou (2019): "Manager Sentiment and Stock Returns"
  - FinBERT (Araci, 2019): BERT fine-tuned on financial text

Sentiment scoring:
  - Management tone → bullish/bearish probability
  - Guidance keywords → forward expectation score
  - Q&A tone shift → analyst skepticism detector
  - Combined score → forecast signal [-20, +20]
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Keyword dictionaries for rule-based fallback sentiment
BULLISH_KEYWORDS = {
    "strong demand", "record revenue", "exceeded expectations", "growth trajectory",
    "robust pipeline", "market share gains", "margin expansion", "double digit growth",
    "accelerating", "outperform", "exceeded guidance", "raised guidance",
    "strong momentum", "healthy demand", "pricing power", "operating leverage",
    "excellent quarter", "beat expectations", "new highs", "record earnings",
    "positive outlook", "upside", "tailwinds", "breakthrough", "unprecedented demand",
}

BEARISH_KEYWORDS = {
    "headwinds", "challenging environment", "softening demand", "margin compression",
    "cautious outlook", "uncertainty", "inventory buildup", "restructuring",
    "weak demand", "below expectations", "missed guidance", "lowered guidance",
    "downside risks", "cost pressures", "supply chain disruption", "slowdown",
    "disappointing", "difficult quarter", "competitive pressure", "pricing pressure",
    "macro concerns", "recessionary", "decline", "deteriorating",
}

FORWARD_POSITIVE = {
    "we expect", "we anticipate growth", "optimistic", "confident in",
    "positive momentum", "raised full year", "increased guidance",
    "on track to deliver", "well positioned", "bright outlook",
}

FORWARD_NEGATIVE = {
    "remain cautious", "uncertain outlook", "reduced expectations",
    "lowered full year", "revised downward", "tempered expectations",
    "risk of slowdown", "less visibility", "conservative approach",
}


@dataclass
class SentimentResult:
    """Result of transcript sentiment analysis."""
    positive_score: float = 0.0   # [0, 1]
    negative_score: float = 0.0   # [0, 1]
    neutral_score: float = 0.0    # [0, 1]
    forward_score: float = 0.0    # [-1, +1]
    composite_score: float = 0.0  # [-1, +1]
    forecast: float = 0.0         # [-20, +20] Carver-format
    confidence: float = 0.0       # [0, 1]
    detail: str = ""


@dataclass
class EarningsNLPResult:
    """Batch result for all transcripts processed."""
    signals: Dict[str, SentimentResult] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


class EarningsCallNLP:
    """NLP analysis of earnings call transcripts.

    Parameters
    ----------
    use_finbert : bool
        Try to use FinBERT transformer model (default True).
        Falls back to keyword-based analysis if unavailable.
    max_chunks : int
        Maximum text chunks to process per transcript (default 50).
    chunk_size : int
        Tokens per chunk for FinBERT (default 512).
    management_weight : float
        Weight for management discussion section (default 0.6).
    qa_weight : float
        Weight for Q&A section (default 0.4).
    """

    def __init__(
        self,
        use_finbert: bool = True,
        max_chunks: int = 50,
        chunk_size: int = 512,
        management_weight: float = 0.6,
        qa_weight: float = 0.4,
    ):
        self.use_finbert = use_finbert
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.management_weight = management_weight
        self.qa_weight = qa_weight
        self._model = None
        self._tokenizer = None

    def _load_finbert(self) -> bool:
        """Attempt to load FinBERT model."""
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            logger.info("FinBERT loaded successfully")
            return True
        except Exception as e:
            logger.warning("FinBERT not available (%s), using keyword fallback", e)
            return False

    def _finbert_sentiment(self, text: str) -> Tuple[float, float, float]:
        """Run FinBERT on text chunk. Returns (positive, negative, neutral)."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.chunk_size,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # FinBERT output: [positive, negative, neutral]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _keyword_sentiment(self, text: str) -> Tuple[float, float, float]:
        """Rule-based keyword sentiment as fallback."""
        text_lower = text.lower()

        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        total = bullish_count + bearish_count + 1  # +1 for smoothing

        pos = bullish_count / total
        neg = bearish_count / total
        neu = 1.0 - pos - neg

        return pos, neg, max(0.0, neu)

    def _forward_guidance_score(self, text: str) -> float:
        """Score forward-looking statements. Returns [-1, +1]."""
        text_lower = text.lower()

        pos_count = sum(1 for kw in FORWARD_POSITIVE if kw in text_lower)
        neg_count = sum(1 for kw in FORWARD_NEGATIVE if kw in text_lower)
        total = pos_count + neg_count

        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _split_sections(self, transcript: str) -> Tuple[str, str]:
        """Split transcript into management discussion and Q&A."""
        # Common section markers
        qa_markers = [
            r"question[\s-]*and[\s-]*answer",
            r"q\s*&\s*a\s*session",
            r"operator.*questions",
            r"we.*now.*open.*questions",
        ]

        text = transcript
        mgmt_section = transcript
        qa_section = ""

        for marker in qa_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                mgmt_section = text[:match.start()]
                qa_section = text[match.start():]
                break

        return mgmt_section, qa_section

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks suitable for model processing."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if len(chunks) >= self.max_chunks:
                break
        return chunks

    def analyze_transcript(self, transcript: str, symbol: str = "") -> SentimentResult:
        """Analyze a single earnings call transcript.

        Parameters
        ----------
        transcript : str
            Full text of earnings call transcript.
        symbol : str
            Stock symbol for reference.

        Returns
        -------
        SentimentResult with composite score and forecast.
        """
        if not transcript or len(transcript) < 100:
            return SentimentResult(detail="Transcript too short")

        # Split into sections
        mgmt_text, qa_text = self._split_sections(transcript)

        # Choose sentiment method
        use_model = self.use_finbert and self._load_finbert()
        sentiment_fn = self._finbert_sentiment if use_model else self._keyword_sentiment

        # Analyze management section
        mgmt_chunks = self._chunk_text(mgmt_text)
        mgmt_pos, mgmt_neg, mgmt_neu = 0.0, 0.0, 0.0
        for chunk in mgmt_chunks:
            p, n, ne = sentiment_fn(chunk)
            mgmt_pos += p
            mgmt_neg += n
            mgmt_neu += ne
        n_mgmt = max(len(mgmt_chunks), 1)
        mgmt_pos /= n_mgmt
        mgmt_neg /= n_mgmt
        mgmt_neu /= n_mgmt

        # Analyze Q&A section
        qa_pos, qa_neg, qa_neu = 0.0, 0.0, 0.0
        if qa_text:
            qa_chunks = self._chunk_text(qa_text)
            for chunk in qa_chunks:
                p, n, ne = sentiment_fn(chunk)
                qa_pos += p
                qa_neg += n
                qa_neu += ne
            n_qa = max(len(qa_chunks), 1)
            qa_pos /= n_qa
            qa_neg /= n_qa
            qa_neu /= n_qa

        # Weighted combination
        w_m = self.management_weight
        w_q = self.qa_weight if qa_text else 0.0
        w_total = w_m + w_q
        if w_total > 0:
            pos = (mgmt_pos * w_m + qa_pos * w_q) / w_total
            neg = (mgmt_neg * w_m + qa_neg * w_q) / w_total
            neu = (mgmt_neu * w_m + qa_neu * w_q) / w_total
        else:
            pos, neg, neu = mgmt_pos, mgmt_neg, mgmt_neu

        # Forward guidance
        forward = self._forward_guidance_score(transcript)

        # Composite score: sentiment + forward guidance
        sentiment_net = pos - neg  # [-1, +1]
        composite = 0.7 * sentiment_net + 0.3 * forward  # [-1, +1]

        # Convert to Carver forecast [-20, +20]
        forecast = composite * 20.0

        # Confidence based on how decisive the sentiment is
        confidence = abs(pos - neg) + abs(forward) * 0.3
        confidence = min(1.0, confidence)

        # Tone shift detection: if Q&A is significantly more negative than mgmt
        tone_shift = (qa_neg - mgmt_neg) if qa_text else 0.0
        if tone_shift > 0.15:  # Analysts skeptical
            forecast *= 0.7
            detail = f"Tone shift detected ({tone_shift:.2f}): analysts more skeptical"
        else:
            detail = f"Mgmt: +{mgmt_pos:.2f}/-{mgmt_neg:.2f}, QA: +{qa_pos:.2f}/-{qa_neg:.2f}"

        method = "FinBERT" if use_model else "keyword"
        return SentimentResult(
            positive_score=round(pos, 4),
            negative_score=round(neg, 4),
            neutral_score=round(neu, 4),
            forward_score=round(forward, 4),
            composite_score=round(composite, 4),
            forecast=round(forecast, 2),
            confidence=round(confidence, 4),
            detail=f"[{method}] {detail}",
        )

    def analyze_batch(
        self,
        transcripts: Dict[str, str],
    ) -> EarningsNLPResult:
        """Analyze multiple transcripts.

        Parameters
        ----------
        transcripts : dict
            {symbol: transcript_text}

        Returns
        -------
        EarningsNLPResult with per-symbol signals.
        """
        result = EarningsNLPResult()

        for symbol, text in transcripts.items():
            try:
                sig = self.analyze_transcript(text, symbol)
                result.signals[symbol] = sig
            except Exception as e:
                logger.warning("Earnings NLP failed for %s: %s", symbol, e)

        n_bullish = sum(1 for s in result.signals.values() if s.forecast > 5)
        n_bearish = sum(1 for s in result.signals.values() if s.forecast < -5)
        result.log.append(
            f"Earnings NLP: {len(result.signals)} transcripts → "
            f"{n_bullish} bullish, {n_bearish} bearish"
        )

        for line in result.log:
            logger.info("EarningsNLP: %s", line)

        return result
