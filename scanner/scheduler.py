from __future__ import annotations
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

_scheduler = None
_lock = threading.Lock()


def get_or_create_scheduler() -> BackgroundScheduler:
    """
    Singleton APScheduler. Safe to call from multiple Streamlit reruns.
    Scans run at 09:45, 11:00, 13:00, 15:00 ET (Mon-Fri).
    """
    global _scheduler
    with _lock:
        if _scheduler is None:
            from scanner.run_scan import run_full_scan
            tz = pytz.timezone('America/New_York')
            _scheduler = BackgroundScheduler(timezone=tz)
            for hour, minute in [(9, 45), (11, 0), (13, 0), (15, 0)]:
                _scheduler.add_job(
                    run_full_scan,
                    trigger='cron',
                    hour=hour,
                    minute=minute,
                    day_of_week='mon-fri',
                    id=f'scan_{hour:02d}{minute:02d}',
                    replace_existing=True,
                )
            _scheduler.start()
            print('[Scheduler] Started. Scans at 09:45, 11:00, 13:00, 15:00 ET (Mon-Fri).')
    return _scheduler
