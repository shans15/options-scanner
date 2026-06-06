from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from domain.technical_signals import TechnicalSetup
from domain.technical_signals import (
    _emas, _atr14, _phase_oscillator, _po_bandwidth_percentile,
)
from domain.technical_signals import _detect_compression_breakout


def test_technical_setup_is_frozen_with_required_fields():
    s = TechnicalSetup(
        setup_name='compression_breakout',
        direction='bullish',
        strength=0.85,
        notes='PO squeeze rank 0.08, ribbon stacked',
    )
    assert s.setup_name == 'compression_breakout'
    assert s.direction == 'bullish'
    assert 0.0 <= s.strength <= 1.0
    with pytest.raises(FrozenInstanceError):
        s.strength = 0.5  # frozen → cannot mutate


def _synth_ohlcv(close_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'open':   close_series.shift(1).fillna(close_series.iloc[0]),
        'high':   close_series * 1.005,
        'low':    close_series * 0.995,
        'close':  close_series,
        'volume': pd.Series(1_000_000, index=close_series.index),
    })


def _ohlcv_from_close(close_vals: list[float]) -> pd.DataFrame:
    close = pd.Series(close_vals)
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })


def test_emas_returns_dict_with_expected_lengths():
    close = pd.Series(np.linspace(100, 110, 250))
    out = _emas(close)
    assert set(out.keys()) == {8, 13, 21, 48, 200}
    for n, ema in out.items():
        assert len(ema) == len(close)
        # last EMA value should be near the last close for a linear series
        assert abs(ema.iloc[-1] - close.iloc[-1]) < 5.0


def test_atr14_is_positive_and_smoothed():
    close = pd.Series(np.linspace(100, 120, 250))
    df = _synth_ohlcv(close)
    atr = _atr14(df['high'], df['low'], df['close'])
    assert (atr.dropna() > 0).all()
    assert len(atr) == len(close)


def test_phase_oscillator_zero_when_close_equals_ema21():
    close = pd.Series([100.0] * 250)
    df = _synth_ohlcv(close)
    emas = _emas(df['close'])
    atr = _atr14(df['high'], df['low'], df['close'])
    po = _phase_oscillator(df['close'], emas[21], atr)
    # On a flat series with positive ATR (from 0.5% wicks), PO should hover near 0
    assert abs(po.iloc[-1]) < 0.5


def test_po_bandwidth_percentile_is_in_unit_range():
    np.random.seed(1)
    po = pd.Series(np.random.normal(0, 1, 300))
    pct = _po_bandwidth_percentile(po)
    # After enough warmup, values should be in [0, 1]
    tail = pct.dropna().tail(50)
    assert ((tail >= 0) & (tail <= 1)).all()


def test_compression_breakout_bullish_fires_on_squeeze_with_stacked_ribbon():
    # Build a 250-bar series: 100 volatile bars establish a noisy background,
    # 30 bars of uptrend stack the EMA ribbon bullishly, then 120 very tight flat
    # bars create a genuine PO squeeze (low percentile rank vs the volatile history).
    np.random.seed(99)
    phase1 = 100 + np.cumsum(np.random.normal(0, 2.0, 100))   # volatile background
    start = phase1[-1]
    phase2 = np.linspace(start, start + 25, 30)               # uptrend → stacks ribbon
    phase3 = np.full(120, phase2[-1]) + np.random.normal(0, 0.005, 120)  # tight squeeze
    close = np.concatenate([phase1, phase2, phase3])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_compression_breakout(df)

    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'compression_breakout'
    assert 0.0 < setup.strength <= 1.0


def test_compression_breakout_returns_none_when_not_squeezed():
    # Volatile, non-squeezed series → no signal
    np.random.seed(11)
    close = 100 + np.cumsum(np.random.normal(0, 1.0, 250))
    df = _ohlcv_from_close(close.tolist())
    assert _detect_compression_breakout(df) is None


def test_compression_breakout_bearish_fires_on_inverse_setup():
    # Mirror of bullish: volatile background, downtrend stacks ribbon bearishly,
    # then tight flat creates a genuine PO squeeze.
    np.random.seed(99)
    phase1 = 100 + np.cumsum(np.random.normal(0, 2.0, 100))   # volatile background
    start = phase1[-1]
    phase2 = np.linspace(start, start - 25, 30)               # downtrend → stacks ribbon bearishly
    phase3 = np.full(50, phase2[-1]) + np.random.normal(0, 0.005, 50)   # tight squeeze
    close = np.concatenate([phase1, phase2, phase3])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_compression_breakout(df)
    assert setup is not None
    assert setup.direction == 'bearish'


from domain.technical_signals import _detect_pullback_in_trend


def test_pullback_in_trend_bullish_fires_on_uptrend_pullback_to_21ema():
    # Build a 242-bar steady uptrend, then two bars that hold near the top while
    # the last bar's WICK dips to touch the 13-21 EMA zone.
    # Fixture deviation from plan: the pullback is in the intrabar wick (low = close * 0.985),
    # NOT a down-close, so the close stays above e200 and the low still hits [e21*0.99, e13*1.01].
    # A 3% close pullback would land the low below the EMA zone on a linear ramp.
    base = np.linspace(80, 130, 240)
    close = np.concatenate([base, [base[-1], base[-1]]])  # hold at top for last 2 bars
    df = pd.DataFrame({
        'open':   pd.Series(close).shift(1).fillna(close[0]),
        'high':   pd.Series(close) * 1.003,
        'low':    pd.Series(close) * 0.985,   # 1.5% wick dips into the 13-21 EMA zone
        'close':  pd.Series(close),
        'volume': pd.Series([1_000_000] * (len(close) - 1) + [600_000]),  # last bar low vol
    })

    setup = _detect_pullback_in_trend(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'pullback_in_trend'


def test_pullback_in_trend_returns_none_on_flat_market():
    close = pd.Series(np.full(250, 100.0) + np.random.RandomState(3).normal(0, 0.5, 250))
    df = _ohlcv_from_close(close.tolist())
    assert _detect_pullback_in_trend(df) is None


def test_pullback_in_trend_bearish_fires_on_downtrend_rally_to_21ema():
    # Mirror of bullish: 240-bar downtrend stacks ribbon bearishly, then two bars
    # hold at the bottom while the last bar's HIGH wick rallies to touch [e13*0.99, e21*1.01].
    # Fixture uses high = close * 1.015; the close stays below e200 so bearish_stack holds.
    base = np.linspace(130, 80, 240)
    close = np.concatenate([base, [base[-1], base[-1]]])  # hold at bottom for last 2 bars
    df = pd.DataFrame({
        'open':   pd.Series(close).shift(1).fillna(close[0]),
        'high':   pd.Series(close) * 1.015,   # 1.5% wick rallies into the 21-13 EMA zone
        'low':    pd.Series(close) * 0.997,
        'close':  pd.Series(close),
        'volume': pd.Series([1_000_000] * (len(close) - 1) + [600_000]),
    })
    setup = _detect_pullback_in_trend(df)
    assert setup is not None
    assert setup.direction == 'bearish'
