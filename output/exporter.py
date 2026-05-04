from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'scans'


def export_scan(results: list, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """
    Write scan results to a timestamped CSV + JSON file and update latest.json.
    output_dir is created if it does not exist.

    latest.json structure: {"scan_time": ISO string, "results": list of dicts}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_name = f'scan_{timestamp}'

    # Write timestamped CSV
    df = pd.DataFrame(results) if results else pd.DataFrame()
    csv_path = output_dir / f'{base_name}.csv'
    df.to_csv(csv_path, index=False)

    # Write timestamped JSON
    json_path = output_dir / f'{base_name}.json'
    json_path.write_text(json.dumps(results, default=str, indent=2))

    # Always update latest.json
    latest_path = output_dir / 'latest.json'
    latest_data = {
        'scan_time': datetime.now().isoformat(),
        'results': results,
    }
    latest_path.write_text(json.dumps(latest_data, default=str, indent=2))
