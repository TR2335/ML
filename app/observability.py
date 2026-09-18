"""Small dependency-free metrics registry exposed in Prometheus text format."""

from collections import Counter
from threading import Lock
from time import monotonic


class Metrics:
    def __init__(self):
        self._lock = Lock()
        self.counters: Counter[str] = Counter()
        self.latency_sum: Counter[str] = Counter()
        self.latency_count: Counter[str] = Counter()

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self.latency_sum[name] += seconds
            self.latency_count[name] += 1

    def timer(self, name: str):
        started = monotonic()

        class Timer:
            def __enter__(inner):
                return inner

            def __exit__(inner, exc_type, exc_value, traceback):
                self.observe(name, monotonic() - started)

        return Timer()

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self.counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, value in sorted(self.latency_sum.items()):
                lines.append(f"{name}_sum {value:.6f}")
                lines.append(f"{name}_count {self.latency_count[name]}")
        return "\n".join(lines) + ("\n" if lines else "")


metrics = Metrics()
