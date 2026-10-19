"""Small helpers to measure memory, so the README can show before/after numbers."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import psutil


def rss_gb() -> float:
    """RAM currently used by this Python process, in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def file_gb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e9


def frame_gb(df: pl.DataFrame) -> float:
    """In-memory size of a Polars DataFrame, in GB."""
    return df.estimated_size("gb")


@contextmanager
def track(label: str):
    """Print time taken and RAM change for a block of code.

    Note: RSS only shows memory at the start and end. For true peak usage, watch the
    RAM gauge in the Kaggle sidebar while the step runs.
    """
    start_t, start_m = time.perf_counter(), rss_gb()
    yield
    print(
        f"[{label}] {time.perf_counter() - start_t:,.1f}s | RAM {start_m:.2f} -> {rss_gb():.2f} GB"
    )
