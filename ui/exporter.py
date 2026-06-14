from __future__ import annotations
import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

from pipeline.run_scan import ScanResult


def _serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def write_scan(result: ScanResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp.strftime('%Y%m%d_%H%M')
    csv_path = output_dir / f'scan_{stamp}.csv'
    json_path = output_dir / f'scan_{stamp}.json'
    latest_path = output_dir / 'latest.json'

    columns = ['ticker', 'strategy', 'option_type', 'strike', 'expiration', 'dte',
               'spot_price', 'bid', 'ask', 'mid', 'volume', 'open_interest',
               'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
               'pop_blended', 'pop_delta', 'pop_bs', 'pop_historical', 'pop_garch_mc',
               'ev', 'max_adverse_loss', 'margin_estimate', 'composite_score', 'label',
               'setup_name', 'setup_direction', 'setup_strength',
               'weekly_trend', 'streak', 'pct_1w', 'pct_2w', 'pct_4w', 'weekly_agree',
               'vix_now', 'vix_regime', 'vix_pct_vs_7d',
               'reason_for', 'reason_against', 'failed_filters']
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for c in result.candidates:
            con = c.contract
            w.writerow([
                con.ticker, c.strategy.name, con.option_type, con.strike,
                con.expiration.isoformat(), con.dte, con.spot_price,
                con.bid, con.ask, con.mid, con.volume, con.open_interest,
                con.implied_volatility, con.delta, con.gamma, con.theta, con.vega,
                c.pop_blended, c.pop_delta, c.pop_bs, c.pop_historical, c.pop_garch_mc,
                c.ev, c.max_adverse_loss, c.margin_estimate, c.composite_score, c.label,
                c.setup_name or '',
                c.setup_direction or '',
                c.setup_strength if c.setup_strength is not None else '',
                c.weekly_trend or '',
                c.consecutive_close_streak if c.consecutive_close_streak is not None else '',
                f"{c.pct_change_1w:.1%}" if c.pct_change_1w is not None else '',
                f"{c.pct_change_2w:.1%}" if c.pct_change_2w is not None else '',
                f"{c.pct_change_4w:.1%}" if c.pct_change_4w is not None else '',
                c.weekly_ribbon_agreement if c.weekly_ribbon_agreement is not None else '',
                c.vix_now if c.vix_now is not None else '',
                c.vix_regime or '',
                f"{c.vix_pct_vs_7d:.4f}" if c.vix_pct_vs_7d is not None else '',
                c.reason_for, c.reason_against, ';'.join(c.filter_result.failed_filters),
            ])

    payload = {
        'timestamp': result.timestamp.isoformat(),
        'candidates': [
            {
                'contract': _serialize(c.contract),
                'strategy': c.strategy.name,
                'pop_blended': c.pop_blended,
                'pop_delta': c.pop_delta, 'pop_bs': c.pop_bs,
                'pop_historical': c.pop_historical, 'pop_garch_mc': c.pop_garch_mc,
                'stress': _serialize(c.stress),
                'ev': c.ev, 'max_adverse_loss': c.max_adverse_loss,
                'margin_estimate': c.margin_estimate,
                'composite_score': c.composite_score, 'label': c.label,
                'setup_name': c.setup_name,
                'setup_direction': c.setup_direction,
                'setup_strength': c.setup_strength,
                'weekly_trend': c.weekly_trend,
                'consecutive_close_streak': c.consecutive_close_streak,
                'pct_change_1w': c.pct_change_1w,
                'pct_change_2w': c.pct_change_2w,
                'pct_change_4w': c.pct_change_4w,
                'weekly_ribbon_agreement': c.weekly_ribbon_agreement,
                'vix_now': c.vix_now,
                'vix_regime': c.vix_regime,
                'vix_pct_vs_7d': c.vix_pct_vs_7d,
                'reason_for': c.reason_for, 'reason_against': c.reason_against,
                'failed_filters': list(c.filter_result.failed_filters),
            }
            for c in result.candidates
        ],
        'skipped': dict(result.skipped),
    }

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    latest_path.write_text(json.dumps(payload, indent=2, default=str))
    return csv_path, json_path
