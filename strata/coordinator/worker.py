"""Single writer worker (spec §12).

One worker thread per Strata node drains the durable op log. Live sessions and the reflection
engine never apply index mutations themselves — they enqueue and let this one worker apply, so
the zvec single-process-exclusive write lock is never contended by two writers.
"""

from __future__ import annotations

import threading

from strata.coordinator.coordinator import WriteCoordinator


class WriterWorker:
    def __init__(self, coordinator: WriteCoordinator, *, idle_wait_s: float = 0.05) -> None:
        self.coordinator = coordinator
        self.idle_wait_s = idle_wait_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="strata-writer", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        wake = self.coordinator._wake
        while not self._stop.is_set():
            applied = self.coordinator.run_until_idle()
            if applied == 0:
                # Wait for an enqueue/retry signal or poll for ops leaving backoff.
                wake.wait(timeout=self.idle_wait_s)
                wake.clear()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self.coordinator._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
