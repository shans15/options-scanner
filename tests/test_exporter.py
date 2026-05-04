from __future__ import annotations
import json
import tempfile
from pathlib import Path
from output.exporter import export_scan

SAMPLE_RESULTS = [
    {
        'ticker': 'SPY', 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 515.0, 'bid': 1.20, 'ask': 1.40, 'mid': 1.30,
        'premium': 1.30, 'delta': -0.15, 'IV': 0.18, 'IV_rank': 65.0,
        'PoP_blended': 0.84, 'expected_value': 1.05, 'composite_score': 72.0,
        'decision': 'TRADE', 'hard_filter_passed': True,
    }
]


def test_export_creates_csv_and_json(tmp_path: Path) -> None:
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    csv_files = list(tmp_path.glob('scan_*.csv'))
    json_files = list(tmp_path.glob('scan_*.json'))
    assert len(csv_files) == 1
    assert len(json_files) == 1


def test_latest_json_written(tmp_path: Path) -> None:
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    latest = tmp_path / 'latest.json'
    assert latest.exists()
    data = json.loads(latest.read_text())
    assert isinstance(data, dict)
    assert 'results' in data
    assert 'scan_time' in data
    assert data['results'][0]['ticker'] == 'SPY'


def test_csv_contains_data(tmp_path: Path) -> None:
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    csv_file = list(tmp_path.glob('scan_*.csv'))[0]
    content = csv_file.read_text()
    assert 'SPY' in content
    assert 'TRADE' in content


def test_empty_results_still_writes_files(tmp_path: Path) -> None:
    export_scan([], output_dir=tmp_path)
    assert (tmp_path / 'latest.json').exists()
    latest_data = json.loads((tmp_path / 'latest.json').read_text())
    assert latest_data['results'] == []


def test_output_dir_created_if_missing(tmp_path: Path) -> None:
    new_dir = tmp_path / 'nested' / 'scans'
    assert not new_dir.exists()
    export_scan(SAMPLE_RESULTS, output_dir=new_dir)
    assert new_dir.exists()
    assert len(list(new_dir.glob('scan_*.csv'))) == 1
