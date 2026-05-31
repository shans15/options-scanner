import numpy as np
import pandas as pd
from domain.signals import Regime, compute_regime


def _make_log_returns(daily_vol: float, n: int = 252, seed: int = 7) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0, daily_vol, size=n))


def _make_chain_with_iv(iv: float) -> list:
    class FakeRaw:
        def __init__(self, iv, dte, strike, spot):
            self.implied_volatility = iv
            self.dte = dte
            self.strike = strike
            self.spot_price = spot
            self.expiration = None
            self.option_type = 'put'
            self.bid = 1.0; self.ask = 1.1; self.volume = 1; self.open_interest = 1
            self.ticker = 'X'
    return [FakeRaw(iv, 30, 100.0, 100.0) for _ in range(5)]


def test_regime_favors_sell_when_iv_exceeds_rv():
    log_returns = _make_log_returns(daily_vol=0.005)
    chain = _make_chain_with_iv(iv=0.30)
    regime = compute_regime('SPY', log_returns, chain)
    assert 'sell' in regime.favored
    assert 'buy' not in regime.favored
    assert regime.rv_iv_ratio < 0.8


def test_regime_favors_buy_when_rv_exceeds_iv():
    log_returns = _make_log_returns(daily_vol=0.025)
    chain = _make_chain_with_iv(iv=0.15)
    regime = compute_regime('SPY', log_returns, chain)
    assert 'buy' in regime.favored
    assert 'sell' not in regime.favored
    assert regime.rv_iv_ratio > 1.2


def test_regime_neutral_when_rv_near_iv():
    log_returns = _make_log_returns(daily_vol=0.0127)  # 0.0126 yields ratio=0.799 (just below 0.8 threshold)
    chain = _make_chain_with_iv(iv=0.20)
    regime = compute_regime('SPY', log_returns, chain)
    assert set(regime.favored) == {'sell', 'buy'}
    assert 0.8 <= regime.rv_iv_ratio <= 1.2


def test_regime_empty_chain_defaults_to_neutral():
    log_returns = _make_log_returns(daily_vol=0.01)
    regime = compute_regime('SPY', log_returns, chain=[])
    assert regime.favored == ['sell', 'buy']
    assert regime.iv_atm == 0.0
