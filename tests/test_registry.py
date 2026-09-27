"""Run registry gate: the cap, the per-run lock, and the sliding expiry. Model-free.

No browser, no model, no server, no sleeping: the clock is injected, so the 60 second expiry is
proved by moving a counter rather than by waiting a minute. This is what makes the registry
testable before the MCP server exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localdecide.registry import (
    FINISHED,
    MAX_CONCURRENT_RUNS,
    RUNNING,
    WAITING,
    RegistryFull,
    RunRegistry,
)


class FakeClock:
    """A clock the test moves by hand. `now` is the only state."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def check_cap_refuses_the_fifth() -> tuple[int, int]:
    """Four runs go in; the fifth is refused, not queued."""
    clock = FakeClock()
    registry = RunRegistry(clock=clock)
    for _ in range(MAX_CONCURRENT_RUNS):
        registry.create("goal")
    assert registry.active_count() == MAX_CONCURRENT_RUNS
    refused = 0
    try:
        registry.create("one too many")
    except RegistryFull as error:
        refused = 1
        assert error.active == MAX_CONCURRENT_RUNS and error.cap == MAX_CONCURRENT_RUNS, vars(error)
    assert refused == 1, "the fifth run must be refused"
    # Refused means refused: the cap is not a queue, and the refused run is not stored.
    assert registry.active_count() == MAX_CONCURRENT_RUNS
    assert len(registry.list()) == MAX_CONCURRENT_RUNS
    return MAX_CONCURRENT_RUNS, refused


def check_finished_slides_and_expires() -> tuple[float, bool]:
    """A finished run survives 60 s of silence, and every call resets that window."""
    clock = FakeClock()
    registry = RunRegistry(clock=clock)
    run = registry.create("goal")
    registry.start(run.id)
    registry.finish(run.id)

    # 59 s of silence: still there.
    clock.advance(59.0)
    assert registry.get(run.id).id == run.id, "a finished run must survive 59 s"

    # Now slide: keep touching it, and it outlives 60 s from the start.
    clock.advance(30.0)
    assert registry.get(run.id).id == run.id
    clock.advance(30.0)          # 119 s since finish, but only 60 s since the last call
    assert registry.get(run.id).id == run.id, "every call must reset the timer"

    # Stop talking and let the full TTL elapse: reaped.
    clock.advance(60.0)
    gone = False
    try:
        registry.get(run.id)
    except KeyError:
        gone = True
    assert gone, "a finished run must be reaped after 60 s of silence"
    return 60.0, gone


def check_waiting_and_running_never_expire() -> tuple[float, bool]:
    """Only finished runs have a timer. A four-minute task is not a leak."""
    clock = FakeClock()
    registry = RunRegistry(clock=clock)
    waiting = registry.create("waiting")
    running = registry.create("running")
    registry.start(running.id)

    clock.advance(3600.0)       # an hour: far longer than any TTL
    assert registry.get(waiting.id).id == waiting.id, "a waiting run must never expire"
    assert registry.get(running.id).id == running.id, "a running run must never expire"
    # And they are still holding the cap, which is the point of not expiring them.
    assert registry.active_count() == 2, registry.active_count()
    return 3600.0, True


def check_finishing_frees_the_cap() -> tuple[int, int]:
    """The cap is released by finishing a run, not by waiting for a timer."""
    clock = FakeClock()
    registry = RunRegistry(clock=clock)
    runs = [registry.create("g") for _ in range(MAX_CONCURRENT_RUNS)]
    for run in runs:
        registry.start(run.id)
    assert registry.active_count() == MAX_CONCURRENT_RUNS

    registry.finish(runs[0].id)
    # A finished run does not count against the cap, so a new run fits immediately.
    assert registry.active_count() == MAX_CONCURRENT_RUNS - 1
    replacement = registry.create("new run")
    assert registry.active_count() == MAX_CONCURRENT_RUNS
    assert replacement.status == WAITING
    return MAX_CONCURRENT_RUNS, registry.active_count()


def check_per_run_lock() -> tuple[bool, bool]:
    """Each run has its own lock, and it is exclusive."""
    registry = RunRegistry(clock=FakeClock())
    a = registry.create("a")
    b = registry.create("b")
    lock_a, lock_b = registry.lock(a.id), registry.lock(b.id)
    assert lock_a is not lock_b, "two runs must not share a lock"

    held = lock_a.acquire(blocking=False)
    assert held, "the lock must be acquirable"
    again = lock_a.acquire(blocking=False)
    assert not again, "a held per-run lock must refuse a second acquisition"
    lock_a.release()
    # b's lock is free while a's is held: the locks are independent.
    free_b = lock_b.acquire(blocking=False)
    assert free_b, "one run's lock must not block another run's"
    lock_b.release()
    return True, True


def main() -> None:
    cap, refused = check_cap_refuses_the_fifth()
    ttl, reaped = check_finished_slides_and_expires()
    hours, alive = check_waiting_and_running_never_expire()
    peak, after = check_finishing_frees_the_cap()
    exclusive, independent = check_per_run_lock()

    print("run registry verification passed")
    print(
        f"cap={cap} fifth_refused={refused} "
        f"finished_ttl={ttl:.0f}s reaped={reaped} "
        f"waiting_running_never_expired_after={hours:.0f}s={alive} "
        f"cap_after_finish={after} "
        f"per_run_lock_exclusive={exclusive} locks_independent={independent}"
    )
    _ = peak


if __name__ == "__main__":
    main()
