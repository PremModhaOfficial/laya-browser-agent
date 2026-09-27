"""Run registry: who is allowed to be running, and when a finished run is forgotten.

The loop owns a browser, the browser owns a GPU, and the GPU holds exactly one checkpoint. So
the number of live runs is a hard resource limit, not a policy preference: four is the cap, and
the fifth is **refused rather than queued**. Queueing would be worse than refusing, because a
caller waiting on a refused run still has to be told, and a caller that retries on a queue
turns one overrun into a storm.

Three states, and they expire differently:

* ``waiting``  - created, not started. Never expires. A run that is queued behind the cap is the
  caller's to start or abandon; forgetting it silently would leak the cap forever.
* ``running``  - started. Never expires. A four-minute task is a normal task, and any timer that
  could reap it would kill real work.
* ``finished`` - done, and reaped on a **sliding** 60 s window that every call resets. A client
  polling ``state`` while it reads the result keeps its run alive; a client that stops talking
  loses it, and the cap frees itself without a janitor.

Memory only, by design. A registry that outlived the process would imply runs that can be
resumed, and resumption of a browser run means restoring a browser.

Thread-safe: the MCP server is one loop but a run may be driven from any thread, and the cap
check plus the insert has to be atomic or two callers both see three and both get in.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

#: Concurrent runs (waiting + running) allowed at once. Tabs are cheap; the checkpoint is not.
MAX_CONCURRENT_RUNS = 4

#: How long a finished run survives with no further calls, in seconds.
FINISHED_TTL_SECONDS = 60.0

WAITING = "waiting"
RUNNING = "running"
FINISHED = "finished"


class RegistryFull(RuntimeError):
    """Raised when a new run would exceed the cap. The caller is expected to surface this."""

    def __init__(self, active: int, cap: int = MAX_CONCURRENT_RUNS) -> None:
        super().__init__(f"run registry is full: {active} active, cap {cap}")
        self.active = active
        self.cap = cap


@dataclass
class Run:
    """One tracked run. ``goal`` and whatever the caller wants to carry live in ``data``."""

    id: str
    goal: str = ""
    status: str = WAITING
    created_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    data: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "data": dict(self.data),
        }


class RunRegistry:
    """The cap, the per-run lock, and the sliding expiry. No model, no browser, no I/O."""

    def __init__(
        self,
        cap: int = MAX_CONCURRENT_RUNS,
        finished_ttl: float = FINISHED_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cap = cap
        self.finished_ttl = finished_ttl
        # Injectable so the expiry test does not sleep for a real minute.
        self._clock = clock
        self._runs: Dict[str, Run] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def create(self, goal: str = "") -> Run:
        """Register a waiting run, or refuse because the cap is reached.

        Refuses rather than queues. The cap counts waiting and running together, because a
        waiting run has already been promised a slot and letting waiting runs accumulate would
        turn the cap into a suggestion.
        """
        with self._guard:
            self._expire_locked()
            active = sum(1 for run in self._runs.values() if run.status in (WAITING, RUNNING))
            if active >= self.cap:
                raise RegistryFull(active, self.cap)
            run = Run(id=uuid.uuid4().hex, goal=goal)
            self._runs[run.id] = run
            self._locks[run.id] = threading.Lock()
            return run

    def start(self, run_id: str) -> Run:
        """Move a waiting run to running. Touches its expiry timer, which waiting runs ignore."""
        with self._guard:
            run = self._require(run_id)
            if run.status == WAITING:
                run.status = RUNNING
            return run

    def finish(self, run_id: str) -> Run:
        """Mark a run finished. From now its lifetime is the sliding TTL."""
        with self._guard:
            run = self._require(run_id)
            run.status = FINISHED
            run.finished_at = self._clock()
            return run

    def abort(self, run_id: str) -> Run:
        """Abandon a run without finishing it. Same reaping as a finish, different cause."""
        run = self.finish(run_id)
        run.data["aborted"] = True
        return run

    # -- access ------------------------------------------------------------

    def get(self, run_id: str) -> Run:
        """Read a run. Every call resets a finished run's TTL - that is what sliding means.

        Calling `state` in a poll loop is the normal way a client watches a run finish, so the
        TTL is reset on read as well as on write. A waiting or running run has no timer at all.
        """
        with self._guard:
            self._expire_locked()
            run = self._require(run_id)
            if run.status == FINISHED:
                run.finished_at = self._clock()
            return run

    def lock(self, run_id: str) -> threading.Lock:
        """The per-run lock, so two callers cannot drive the same run at once."""
        with self._guard:
            self._require(run_id)
            return self._locks[run_id]

    def list(self) -> List[Run]:
        """Every live run, oldest first, after reaping anything expired."""
        with self._guard:
            self._expire_locked()
            return sorted(self._runs.values(), key=lambda run: run.created_at)

    def active_count(self) -> int:
        """Waiting plus running. The number the cap is measured against."""
        with self._guard:
            return sum(1 for run in self._runs.values() if run.status in (WAITING, RUNNING))

    # -- internals ---------------------------------------------------------

    def _require(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        return run

    def _expire_locked(self) -> None:
        """Reap finished runs past their TTL. Waiting and running runs are never touched.

        Called with the guard held, on every public operation, so there is no janitor thread and
        no window where a stale run is still counted against the cap.
        """
        now = self._clock()
        for run_id, run in list(self._runs.items()):
            if run.status != FINISHED or run.finished_at is None:
                continue
            if now - run.finished_at >= self.finished_ttl:
                del self._runs[run_id]
                self._locks.pop(run_id, None)
