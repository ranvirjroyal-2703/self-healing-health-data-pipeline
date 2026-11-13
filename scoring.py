"""Cost model for sandbox runs. Lower cost is better."""

from sandbox_runner import SandboxResult


_MEMORY_MB = {"128m": 128, "256m": 256, "512m": 512}


def memory_mb(limit: str) -> int:
    return _MEMORY_MB.get(limit, 256)


def run_cost(result: SandboxResult, memory_limit: str) -> float:
    """
    Estimated cost of one sandbox attempt.

    Successful runs: seconds * memory factor.
    Failures are heavily penalized so they never outrank a real clean.
    Containment (timeout / OOM kill) is recorded but is not a winning outcome.
    """
    mem = memory_mb(memory_limit) / 256.0
    base = result.duration_sec * mem
    if result.success and not result.timed_out:
        return round(base, 3)
    penalty = 100.0 if (result.timed_out or result.killed_by_limit) else 50.0
    return round(penalty + base, 3)
