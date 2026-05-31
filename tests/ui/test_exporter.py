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
