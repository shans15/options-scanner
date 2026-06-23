from __future__ import annotations
from domain.stock_scanner.types import StockSetup, StockCandidate

# Per-detector weights (sum to 1.0)
_WEIGHTS = {
    "sma_stack": 0.30,
    "compression": 0.25,
    "proximity_52w": 0.25,
    "volume_surge": 0.20,
}


def score_candidate(ticker: str, spot: float, setups: list[StockSetup], snapshot: dict) -> StockCandidate:
    """Aggregate per-detector setups into a candidate with bullish/bearish/net scores."""
    bullish = sum(_WEIGHTS[s.setup_type] * s.strength for s in setups if s.direction == "bullish")
    bearish = sum(_WEIGHTS[s.setup_type] * s.strength for s in setups if s.direction == "bearish")
    bullish_score = round(bullish * 100, 1)
    bearish_score = round(bearish * 100, 1)
    net = bullish_score - bearish_score

    if net >= 60:
        label = "STRONG_BULL"
    elif net >= 30:
        label = "BULL"
    elif net <= -60:
        label = "STRONG_BEAR"
    elif net <= -30:
        label = "BEAR"
    else:
        label = "NEUTRAL"

    return StockCandidate(
        ticker=ticker, spot=spot, setups=list(setups),
        bullish_score=bullish_score, bearish_score=bearish_score,
        net_score=round(net, 1), label=label, snapshot=snapshot,
    )
