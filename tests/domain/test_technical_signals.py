from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from domain.technical_signals import TechnicalSetup
from domain.technical_signals import (
    _emas, _atr14, _phase_oscillator, _po_bandwidth_percentile,
)
from domain.technical_signals import _detect_compression_breakout
from domain.technical_signals import _detect_pullback_in_trend
from domain.technical_signals import _detect_stage_2_breakout
from domain.technical_signals import _detect_failed_breakdown_reversal


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


def test_stage_2_breakout_bullish_fires_on_fresh_ema48_cross_with_prior_squeeze():
    # Fixture deviation from plan: the plan's simple linspace fixture didn't produce
    # sufficient contrast between the squeeze and background PO stdev (compression
    # percentile stayed ~0.67, above the 0.25 cutoff), and close was already far above
    # e48 so no fresh cross occurred. Redesigned with a volatile background within the
    # 126-bar percentile window, followed by a tight squeeze and a fresh breakout that
    # (a) crosses e48 from below within 5 bars, (b) goes to a new high (near_high=True),
    # and (c) shows recent bbw_pct min < 0.25.
    np.random.seed(42)
    early = np.linspace(50, 80, 150)
    vol_section = np.zeros(70)
    vol_section[0] = 80
    for i in range(1, 70):
        vol_section[i] = vol_section[i - 1] + np.random.normal(0, 4.0)
    vol_section = 80 + (vol_section - vol_section.mean()) * 0.7
    vol_section = np.clip(vol_section, 60, 90)
    plateau = vol_section[-1]
    squeeze = np.full(40, plateau) + np.random.normal(0, 0.02, 40)
    # Determine e48 level and prior high to anchor the breakout
    temp_close = pd.Series(np.concatenate([early, vol_section, squeeze]))
    e48_level = _emas(temp_close)[48].iloc[-1]
    prior_high = (temp_close * 1.003).max()
    breakout = np.linspace(e48_level * 0.99, prior_high * 1.02, 6)
    close = np.concatenate([early, vol_section, squeeze, breakout])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_stage_2_breakout(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'stage_2_breakout'


def test_stage_2_breakout_returns_none_when_far_from_52w_high():
    close = np.concatenate([np.linspace(80, 200, 100), np.linspace(200, 110, 150)])
    df = _ohlcv_from_close(close.tolist())
    assert _detect_stage_2_breakout(df) is None


def test_stage_2_breakout_bearish_fires_on_stage_4_breakdown():
    # Fixture deviation from plan: same reasoning as the bullish case — the plan's
    # linspace fixture didn't create enough PO-stdev contrast for the compression check,
    # and close was already far below e48 so no fresh cross occurred. Redesigned with
    # a volatile background, tight squeeze, and a fresh breakdown that crosses e48 from
    # above within 5 bars and hits a new 52w low (near_low=True).
    np.random.seed(17)
    early = np.linspace(120, 90, 150)
    vol_section = np.zeros(70)
    vol_section[0] = 90
    for i in range(1, 70):
        vol_section[i] = vol_section[i - 1] + np.random.normal(-0.1, 4.0)
    vol_section = 90 + (vol_section - vol_section.mean()) * 0.7
    vol_section = np.clip(vol_section, 70, 110)
    plateau = vol_section[-1]
    squeeze = np.full(40, plateau) + np.random.normal(0, 0.02, 40)
    # Determine e48 level and prior low to anchor the breakdown
    temp_close = pd.Series(np.concatenate([early, vol_section, squeeze]))
    e48_level = _emas(temp_close)[48].iloc[-1]
    prior_low = (temp_close * 0.997).min()
    breakdown = np.linspace(e48_level * 1.005, prior_low * 0.98, 6)
    close = np.concatenate([early, vol_section, squeeze, breakdown])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_stage_2_breakout(df)
    assert setup is not None
    assert setup.direction == 'bearish'


def test_failed_breakdown_reversal_bullish_fires_on_reclaim_with_divergence():
    # Fixture redesign: plan's volume concat had 246 entries vs 245 for close/high/low.
    # Fix: use np.full(244, ...) + [1_800_000] so volume length matches the 245-bar series.
    # Thresholds unchanged; only the array-length bug in the plan fixture is corrected.
    base = np.linspace(120, 100, 240).tolist()
    breakdown_lows = [99.0, 97.5, 95.0, 97.0, 100.5]
    close = np.array(base + breakdown_lows)
    high = np.array(base + [c + 1.0 for c in breakdown_lows])
    low = np.array(base + [c - 0.5 for c in breakdown_lows])
    low[-3] = 93.0
    volume = np.concatenate([
        np.full(244, 1_000_000),
        np.array([1_800_000]),
    ])
    df = pd.DataFrame({
        'open': pd.Series(close).shift(1).fillna(close[0]),
        'high': high, 'low': low, 'close': close, 'volume': volume,
    })

    setup = _detect_failed_breakdown_reversal(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'failed_breakdown_reversal'


def test_failed_breakdown_reversal_returns_none_on_continuous_decline():
    close = np.linspace(120, 80, 250)
    df = _ohlcv_from_close(close.tolist())
    assert _detect_failed_breakdown_reversal(df) is None


def test_failed_breakdown_reversal_bearish_fires_on_failed_breakout():
    # Fixture redesign: same array-length bug as the bullish fixture — plan's volume
    # concat produced 246 entries. Fixed to np.full(244, ...) + [1_800_000] = 245.
    base = np.linspace(80, 100, 240).tolist()
    breakout_highs = [101.0, 102.5, 105.0, 102.0, 99.0]
    close = np.array(base + breakout_highs)
    high = np.array(base + [c + 1.0 for c in breakout_highs])
    high[-3] = 107.0
    low = np.array(base + [c - 0.5 for c in breakout_highs])
    volume = np.concatenate([np.full(244, 1_000_000), np.array([1_800_000])])
    df = pd.DataFrame({
        'open': pd.Series(close).shift(1).fillna(close[0]),
        'high': high, 'low': low, 'close': close, 'volume': volume,
    })

    setup = _detect_failed_breakdown_reversal(df)
    assert setup is not None
    assert setup.direction == 'bearish'
