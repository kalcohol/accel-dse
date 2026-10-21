"""0.316: freeze-aware elapsed time for the suite's timing guards (test_core_184 fuzz watchdog, 099 / 209 / 210 bounds).

Problem (0.315 suite, rt315a): the host froze the whole VM (virtio balloon); every concurrently running test file gained
≈ 100 s of wall time while executing nothing, and a 120 s wall watchdog reported a HANG that was not one.

Mechanism: a daemon sampler thread sleeps PERIOD_S and records each wake-up gap together with this process's CPU time
(time.process_time, every thread) over the gap.  A gap counts as *frozen* only when it is ≥ GAP_S **and** the process
burned < CPU_FRAC of it — i.e. the process was neither running nor able to wake (host/VM freeze, SIGSTOP, total
descheduling).  Frozen time is subtracted; everything else counts in full, so the guards still detect what they did:
  • a blocking / sleeping hang — the sampler keeps waking every PERIOD_S, no gap, counted 1:1 as before;
  • a CPU-bound hang, including a long C call holding the GIL — the gap shows CPU ≈ gap, not frozen, counted 1:1;
  • a compute regression — unchanged (the process is running, so its time is never subtracted).
Only time in which the process did not execute at all is excused.  Thresholds are 假设 (choice, not fitted)."""
from __future__ import annotations

import threading
import time

PERIOD_S = 0.05      # 假设: sampler cadence
GAP_S = 0.5          # 假设: a wake-up ≥ 10 periods late is a stall candidate
CPU_FRAC = 0.2       # 假设: < 20 % CPU over the stall ⇒ the process was not running ⇒ frozen


class FreezeClock:
    def __init__(self, period_s: float = PERIOD_S, gap_s: float = GAP_S, cpu_frac: float = CPU_FRAC):
        self.period_s, self.gap_s, self.cpu_frac = period_s, gap_s, cpu_frac
        self._lock = threading.Lock()
        self._frozen = 0.0
        self._stalls: list[tuple[float, float, float]] = []        # (t_end, gap_s, cpu_s)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="freeze-clock", daemon=True)
        self._t.start()

    def _run(self) -> None:
        w, c = time.monotonic(), time.process_time()
        while not self._stop.wait(self.period_s):
            w1, c1 = time.monotonic(), time.process_time()
            gap, cpu = w1 - w, c1 - c
            if gap >= self.gap_s and cpu < self.cpu_frac * gap:
                with self._lock:
                    self._frozen += gap - self.period_s
                    self._stalls.append((w1, gap, cpu))
            w, c = w1, c1

    def now(self) -> float:
        """Monotonic seconds minus frozen seconds observed so far (use differences only)."""
        with self._lock:
            return time.monotonic() - self._frozen

    def frozen(self) -> float:
        with self._lock:
            return self._frozen

    def stalls(self) -> list[tuple[float, float, float]]:
        with self._lock:
            return list(self._stalls)

    def stop(self) -> None:
        self._stop.set()


_CLOCK: FreezeClock | None = None


def clock() -> FreezeClock:
    """Process-wide shared sampler (started on first use)."""
    global _CLOCK
    if _CLOCK is None:
        _CLOCK = FreezeClock()
    return _CLOCK


if __name__ == "__main__":          # self-check helper used by test_core_316: python3 freeze_clock.py sleep|spin S
    import sys
    mode, s = sys.argv[1], float(sys.argv[2])
    fc = clock()
    time.sleep(0.2)
    w0, a0 = time.monotonic(), fc.now()
    if mode == "sleep":
        time.sleep(s)
    elif mode == "gil":                                 # one C call holding the GIL (no bytecode boundary)
        sum(range(int(s * 1e8)))
    else:
        t_end = time.process_time() + s
        while time.process_time() < t_end:
            pass
    time.sleep(0.2)
    print(f"wall={time.monotonic() - w0:.3f} active={fc.now() - a0:.3f} frozen={fc.frozen():.3f}", flush=True)
