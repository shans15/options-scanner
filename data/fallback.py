from __future__ import annotations
from typing import Any
from data.sources.base import DataSource


class DataFetchError(Exception):
    pass


def fetch_with_fallback(sources: list[DataSource], method: str, *args, **kwargs) -> Any:
    last_err: Exception | None = None
    for src in sources:
        try:
            return getattr(src, method)(*args, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise DataFetchError(f"All sources failed for {method}{args}: {last_err}")
