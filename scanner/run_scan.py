from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np
import yfinance as yf

from scanner.screener import screen_universe
from scanner.chain_fetcher import fetch_contracts
from scanner.quant_engine import run_quant_engine
from scanner.risk_filters import apply_hard_filters
from scanner.scorer import build_scan_result
from output.exporter import export_scan

RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.053'))
MONTE_CARLO_PATHS = int(os.getenv('MONTE_CARLO_PATHS', '10000'))
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'scans'


def run_full_scan(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list:
    """
    Full pipeline: screen -> fetch chains -> quant engine -> filters -> score -> export.
    Returns list of ScanResult dicts sorted by composite_score descending (TRADE first).
    """
    print("[Scan] Starting stock screen...")
    candidates = screen_universe()
    print(f"[Scan] {len(candidates)} candidates after screening.")

    all_results = []

    for ts in candidates:
        ticker = ts.ticker
        spot_price = ts.current_price

        # Fetch 1-year of daily returns for quant models
        try:
            hist = yf.download(ticker, period='1y', progress=False)['Close'].squeeze()
            log_returns = np.log(hist / hist.shift(1)).dropna()
        except Exception:
            continue

        # Fetch option contracts from Robinhood
        contracts = fetch_contracts(ticker, spot_price)
        if not contracts:
            continue

        for contract in contracts:
            try:
                pop_result = run_quant_engine(
                    contract=contract,
                    log_returns=log_returns,
                    risk_free_rate=RISK_FREE_RATE,
                    n_paths=MONTE_CARLO_PATHS,
                )
                filter_result = apply_hard_filters(contract, pop_result)
                scan_result = build_scan_result(
                    contract=contract,
                    pop_result=pop_result,
                    hard_filter_passed=filter_result.passed,
                    filter_reason=filter_result.reason,
                )
                all_results.append(scan_result)
            except Exception as e:
                print(f"[Scan] Error processing {ticker}: {e}")
                continue

    # Sort: TRADE first, then WATCHLIST, then NO TRADE; within each by composite_score desc
    decision_order = {'TRADE': 0, 'WATCHLIST': 1, 'NO TRADE': 2}
    all_results.sort(
        key=lambda x: (
            decision_order.get(x.get('decision', 'NO TRADE'), 2),
            -x.get('composite_score', 0)
        )
    )

    print(f"[Scan] Complete. {len(all_results)} contracts evaluated.")
    export_scan(all_results, output_dir=output_dir)
    return all_results
