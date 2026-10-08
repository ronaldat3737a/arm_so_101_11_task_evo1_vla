"""Peak RAM / VRAM measurement used by every benchmark (RAM is a hard 12 GB constraint on the target machine)."""

from __future__ import annotations

import threading
import time

import psutil


class ResourceMonitor:
    """Context manager sampling process RSS (incl. children) every `interval` s, plus torch CUDA peak memory."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_rss_gb = 0.0
        self.peak_vram_gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _rss(self) -> float:
        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**30

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_gb = max(self.peak_rss_gb, self._rss())
            time.sleep(self.interval)

    def __enter__(self) -> ResourceMonitor:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.peak_rss_gb = max(self.peak_rss_gb, self._rss())
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
        except ImportError:
            pass

    def summary(self) -> dict:
        return {"peak_rss_gb": round(self.peak_rss_gb, 3), "peak_vram_alloc_gb": None if self.peak_vram_gb is None else round(self.peak_vram_gb, 3)}
