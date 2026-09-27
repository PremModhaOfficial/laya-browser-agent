"""The client-facing half of one engine: answer, observe, state, stop.

Plan steps 5 and 6. The loop in `loop.py` already has all the guards; what was missing was the
way a client talks to it. A browser run cannot finish on its own when a value genuinely is not
knowable - a form asks for something only the user knows, the provider has nothing, and the run
sits there. That is what `answer` is: the client supplies the missing value and the run continues
from the same place, with the same guards, having skipped nothing.

Two rules make this safe rather than convenient:

* **Nothing the client types becomes a selector, a coordinate or a decision.** An answer is a
  *value for a named field*, and the loop matches it to a field it has already observed, using
  the same index the model would have used. The model never sees the string and never chooses
  from it.
* **A skip is a first-class answer.** Leaving an optional field empty is a legitimate response,
  so the client can say "no value for this one, and that is correct" without the harness
  treating it as a failure and re-asking forever.

`Session` wraps one `BrowserDecider` and exposes it as four verbs. It owns no decisions of its
own, so a run driven through the session and a run driven by calling the loop directly produce
the same actions - the parity gate already pins the underlying engine, and this adds no path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .loop import BrowserDecider, Run

#: What a client may say about a field it has no value for.
SKIP = "skip"


class UnknownField(KeyError):
    """The answer names a field this session has not offered. Never guessed at."""


@dataclass
class Pending:
    """A field the run is waiting on, and the turn that asked for it.

    `n` and `target` are the loop's own numbers. Holding them is what lets an answer be applied
    to exactly the turn that asked, instead of being matched loosely by label later.
    """

    label: str
    target: str
    n: int
    goal: str
    optional: bool = False


@dataclass
class Session:
    """One client conversation with the engine.

    A session holds the driver, the goal and the loop's accumulated state, so `run` drives to
    completion and `step` continues the same run. The registry from `registry.py` is not wired
    in here on purpose: the cap belongs to the server that owns many sessions, and a session
    that quietly enforced it would make the policy invisible at the tool surface.
    """

    loop: BrowserDecider
    driver: Any
    goal: str = ""
    _run: Optional[Run] = None
    pending: Optional[Pending] = None
    answered: Dict[str, str] = field(default_factory=dict)
    skipped: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        # Every answer this session supplied, and every field the client chose to leave empty,
        # is a value the text provider should now return. The provider is the only component
        # allowed to produce text, so injecting here is what keeps a model-written string out
        # of the decision path - the loop still only ever picks an index.
        provider = self.loop.text_provider
        if callable(provider) and hasattr(provider, "supply"):
            provider.supply(self.answered)

    # -- verbs -------------------------------------------------------------

    def start(self) -> Run:
        """Launch the run. Same engine as `run` and `step`; the registry will wrap this."""
        with self._lock:
            self._run = self.loop.start(self.driver, self.goal)
            return self._run

    def run(self) -> Run:
        """Drive to completion, one turn at a time, pausing only where a value is missing.

        It steps rather than calling the loop's own `run` so that every turn passes through
        `_capture_pending`. A single call to the loop would perform every turn inside one
        frame, and the session would learn it was blocked only by reading the finished run -
        too late to answer anything.
        """
        self.loop._reset()
        with self._lock:
            self._run = Run(goal=self.goal)
        return self._drive_to_completion()

    def step(self) -> Run:
        """Advance exactly one turn. Never closes the driver - the session owns it."""
        with self._lock:
            if self._run is None:
                self._run = Run(goal=self.goal)
            turn = self.loop.step(self.driver, self.goal)
            self._run.steps.extend(turn.steps)
            if turn.stopped not in ("stepped", ""):
                self._run.stopped = turn.stopped
                self._run.error = turn.error
            self._capture_pending(turn)
            return self._run

    def observe(self) -> Dict[str, Any]:
        """The page as the engine sees it, with nothing decided. Safe to call at any time."""
        observation = self.driver.observe()
        return {
            "url": observation.get("url", ""),
            "title": observation.get("title", ""),
            "text": observation.get("text", ""),
            "state_hash": observation.get("state_hash"),
            "elements": [
                {"label": str(action.get("label", "")), "kind": str(action.get("kind", ""))}
                for action in observation.get("actions", []) or []
            ],
        }

    def state(self) -> Dict[str, Any]:
        """Where the run is, in one read: status, steps so far, and anything it is waiting on."""
        with self._lock:
            run = self._run or Run(goal=self.goal)
            return {
                "goal": run.goal,
                "stopped": run.stopped or ("running" if self.loop._number else ""),
                "steps": len(run.steps),
                "last": (
                    {"n": run.steps[-1].n, "operation": run.steps[-1].operation,
                     "label": run.steps[-1].label, "detail": run.steps[-1].detail}
                    if run.steps else None
                ),
                "waiting_for": (
                    {"field": self.pending.label, "optional": self.pending.optional,
                     "n": self.pending.n}
                    if self.pending else None
                ),
                "answered": dict(self.answered),
                "skipped": sorted(self.skipped),
            }

    def stop(self) -> Dict[str, Any]:
        """Stop here and release the driver. Idempotent: stopping twice is not an error."""
        with self._lock:
            self._run = self._run or Run(goal=self.goal)
            if not self._run.stopped:
                self._run.stopped = "stopped"
            try:
                self.driver.close()
            except Exception:
                pass
            return {"stopped": self._run.stopped, "steps": len(self._run.steps)}

    # -- answer ------------------------------------------------------------

    def answer(self, field_label: str, value: str = "", *, skip: bool = False) -> Dict[str, Any]:
        """Supply the value for a field the run is waiting on, or skip it.

        `field_label` must be a field this run actually offered. An answer for anything else is
        refused with `UnknownField` rather than matched loosely, because a value attached to the
        wrong control is worse than no value at all. `skip=True` is the optional-field path: the
        client is saying the field is correctly left empty.
        """
        with self._lock:
            if self.pending is None:
                raise UnknownField("the run is not waiting on a value")
            if not self._matches(field_label):
                raise UnknownField(f"not the field this run asked for: {self.pending.label!r}")
            if skip:
                self.skipped.add(self.pending.label)
                self.pending = None
                return {"skipped": field_label}
            if not value.strip():
                # An empty answer is a skip, not a value. Treating it as a value would put an
                # empty string into a field the run then believes it filled.
                self.skipped.add(self.pending.label)
                self.pending = None
                return {"skipped": field_label}
            self.answered[self.pending.label] = value
            provider = self.loop.text_provider
            if callable(provider) and hasattr(provider, "supply"):
                provider.supply(self.answered)
            self.pending = None
            return {"answered": field_label}

    # -- internals ---------------------------------------------------------

    def _matches(self, field_label: str) -> bool:
        """Match the client's label against the field we asked about, tolerating decoration.

        The loop's labels carry ` | field=...` and ` | keywords=...` added by the observation
        layer, so a client repeating the visible label still matches. This is a comparison
        against one known field, not a search, so it cannot attach a value to the wrong control.
        """
        wanted = self.pending.label if self.pending else ""
        visible = wanted.split(" | ", 1)[0].strip().lower()
        return field_label.strip().lower() == visible

    def _capture_pending(self, turn: Run) -> None:
        """Notice a turn that wanted a value the provider would not give.

        This is the only place a session learns it is blocked: the loop records the refusal as
        a step, and the provider reports nothing. Reading the step is honest - it is the same
        evidence the caller would read - rather than adding a second signal the loop would have
        to maintain.
        """
        for step in turn.steps:
            if step.operation == "TYPE_TEXT" and "text provider returned nothing" in step.detail:
                self.pending = Pending(
                    label=step.label, target=step.target or "", n=step.n, goal=self.goal)
                return
        if turn.stopped in ("done", "max_steps", "blocked", "error"):
            self.pending = None

    def _drive_to_completion(self) -> Run:
        """Keep stepping until the run ends or it needs a value we do not have."""
        while self._run is not None and not self._run.stopped:
            if self.pending is not None:
                break
            before = len(self._run.steps)
            self.step()
            if self._run.stopped or len(self._run.steps) == before:
                break
        return self._run or Run(goal=self.goal)


__all__ = ["Session", "Pending", "UnknownField", "SKIP"]
