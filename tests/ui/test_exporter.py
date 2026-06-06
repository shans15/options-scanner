from datetime import date, datetime
import json
from pathlib import Path
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut
from engine.stress import StressResult
from engine.risk_filters import FilterResult
from engine.scorer import ScoredCandidate
from pipeline.run_scan import ScanConfig, ScanResult
from ui.exporter import write_scan


def _make_candidate():
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.0, ask=2.1, mid=2.05,
        volume=1000, open_interest=2000, implied_volatility=0.18,
        delta=-0.22, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=635.0,
    )
    return ScoredCandidate(
        contract=c, strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1.0, -3.0, -2.0), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='r1', reason_against='',
    )


def test_write_scan_creates_csv_and_json(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={'BAD': 'fetch_failed: x'})
    csv_path, json_path = write_scan(result, tmp_path)
    assert csv_path.exists()
    assert json_path.exists()
    assert (tmp_path / 'latest.json').exists()


def test_write_scan_csv_has_expected_columns(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={})
    csv_path, _ = write_scan(result, tmp_path)
    header = csv_path.read_text().splitlines()[0]
    for col in ('ticker', 'strategy', 'strike', 'expiration', 'mid', 'pop_blended', 'ev', 'composite_score', 'label'):
        assert col in header


def test_write_scan_json_roundtrips(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={})
    _, json_path = write_scan(result, tmp_path)
    data = json.loads(json_path.read_text())
    assert data['candidates'][0]['contract']['ticker'] == 'SPY'
    assert data['skipped'] == {}


def test_exporter_includes_setup_columns_in_csv(tmp_path):
    from datetime import datetime, date
    from engine.scorer import ScoredCandidate
    from engine.stress import StressResult
    from engine.risk_filters import FilterResult
    from domain.contract import Contract
    from domain.strategy import LongCall
    from pipeline.run_scan import ScanResult, ScanConfig
    from ui.exporter import write_scan

    contract = Contract(
        ticker='AAPL', expiration=date(2026, 7, 18), strike=200.0, option_type='call',
        bid=1.0, ask=1.2, mid=1.1, volume=500, open_interest=1000,
        implied_volatility=0.22, delta=0.50, gamma=0.05, theta=-0.04, vega=0.10,
        dte=21, spot_price=200.0,
    )
    candidate = ScoredCandidate(
        contract=contract, strategy=LongCall(),
        pop_blended=0.55, pop_delta=0.5, pop_bs=0.55, pop_historical=0.55, pop_garch_mc=0.55,
        stress=StressResult(stress_1sd=-0.5, stress_2sd=-1.0, stress_expiry=-1.5),
        ev=0.2, max_adverse_loss=1.0, margin_estimate=110.0,
        filter_result=FilterResult(passed=True, failed_filters=[]),
        composite_score=68.0, label='TRADE',
        reason_for='test', reason_against='',
        setup_name='compression_breakout', setup_direction='bullish', setup_strength=0.82,
    )
    result = ScanResult(timestamp=datetime.now(), config=ScanConfig(),
                        candidates=[candidate], skipped={})
    csv_path, _ = write_scan(result, tmp_path)

    text = csv_path.read_text()
    assert 'setup_name' in text
    assert 'compression_breakout' in text
