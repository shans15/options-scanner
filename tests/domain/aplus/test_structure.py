from domain.aplus.types import FeatureScores, TradeStructure
from domain.aplus.structure import select_structure


def _fs(**overrides) -> FeatureScores:
    base = {
        'vol_iv_percentile': 8.0,
        'cat_post_earnings_drift': 0.0,
        'liq_bid_ask_spread': 6.0,
        'macro_sector_rotation': 6.0,
        'vol_vix_regime': 7.0,
    }
    base.update(overrides)
    return FeatureScores(values=base)


def test_high_iv_returns_debit_spread():
    # vol_iv_percentile <=3 means high IV (raw IV >50%); should return debit spread
    fs = _fs(vol_iv_percentile=2.0)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.DEBIT_SPREAD
    assert 'IV' in rationale


def test_post_earnings_drift_with_tight_spreads_returns_long_premium():
    fs = _fs(vol_iv_percentile=8.0, cat_post_earnings_drift=10.0, liq_bid_ask_spread=9.5)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.LONG_PREMIUM


def test_sector_leader_supportive_vol_returns_long_premium():
    fs = _fs(macro_sector_rotation=9.5, vol_vix_regime=8.0)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.LONG_PREMIUM


def test_default_returns_debit_spread():
    fs = _fs()  # no special conditions met
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.DEBIT_SPREAD
