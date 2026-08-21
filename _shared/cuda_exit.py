"""Windows CUDA / PyTorch process-exit helpers.

RTX 50xx + WDDM often abort during interpreter atexit CUDA teardown
(exit code 3221226505 = 0xC0000409 STATUS_STACK_BUFFER_OVERRUN, or
3221225477 = 0xC0000005 ACCESS_VIOLATION). Training itself is fine;
SUMMARY.json is already on disk.

Rules:
  - Prefer ``os._exit`` and do **not** call ``cuda.synchronize`` /
    ``empty_cache`` / ``gc.collect`` right before exit (those trigger the bug).
  - Launchers treat known teardown codes as success when SUMMARY exists.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Common Windows abnormal exits seen after successful CUDA training.
CUDA_TEARDOWN_EXIT_CODES = frozenset(
    {
        3221226505,  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN / fastfail
        3221225477,  # 0xC0000005 ACCESS_VIOLATION
        -1073740791,  # signed 0xC0000409
        -1073741819,  # signed 0xC0000005
    }
)


def is_cuda_teardown_exit(code: int | None) -> bool:
    if code is None:
        return False
    try:
        return int(code) in CUDA_TEARDOWN_EXIT_CODES
    except (TypeError, ValueError):
        return False


def train_succeeded(returncode: int | None, summary_paths: Iterable[Path] | bool) -> bool:
    """True if child exited 0, or tore down after writing SUMMARY."""
    if isinstance(summary_paths, bool):
        has_summary = summary_paths
    else:
        has_summary = any(True for _ in summary_paths)
    if returncode == 0:
        return True
    return bool(has_summary) and is_cuda_teardown_exit(returncode)


def normalize_train_rc(returncode: int | None, summary_paths: Iterable[Path] | bool) -> int:
    """Map CUDA-teardown-after-SUMMARY to 0 for logging / meta."""
    if train_succeeded(returncode, summary_paths):
        return 0
    return int(returncode or 1)


def exit_clean(code: int = 0) -> None:
    """Hard-exit; skip PyTorch/CUDA atexit destructors that crash on Windows."""
    # Intentionally no torch.cuda.synchronize / empty_cache / gc.collect.
    os._exit(int(code))
