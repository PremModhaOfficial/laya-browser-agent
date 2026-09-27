"""The loop: observe, decide once, act, repeat - with the guardrails that keep it honest.

Two rules make a small decision model safe to run unattended against a real browser:

1. **The model picks an index, your code does everything else.** The decision names
   `operation` and a target index; the executor resolves that index to its own node
   handle. Model output never becomes a selector, a coordinate, or executable code.
2. **The loop, not the model, tracks history.** Repeated actions, unchanged pages and
   step budgets are detected in code, so a model that loops is stopped by the harness
   rather than trusted to notice.

The loop is driver-agnostic on purpose: pass any object with
`observe() -> observation` and `execute(operation, element, text) -> result`.
`localdecide.drivers` ships a Playwright driver; a CDP driver is about 40 lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .decider import Decider
from .page import TARGETED_OPERATIONS, build_element_table, drop_tried_options, table_to_questions
from .scope import Scope


class Driver(Protocol):
    """What the loop needs from a browser. Implement these three things and you are done."""

    def observe(self) -> Dict[str, Any]:
        """Return the current observation (url, title, text, actions/elements)."""
        ...

    def execute(self, operation: str, element: Optional["ElementRef"], text: Optional[str] = None) -> Dict[str, Any]:
        """Perform the operation. Return {"ok": bool, "detail": str, "page_changed": bool}.

        `text` carries the payload for the operations that need one:

        * `TYPE_TEXT` - the string to enter
        * `SELECT`    - the option *value* to choose (already resolved from the observed
                        option list by the loop, so the driver never guesses)

        It is None for every other operation.
        """
        ...

    def close(self) -> None:
        ...


@dataclass
class ElementRef:
    """The loop hands your executor the element it must act on."""

    index: str
    label: str
    role: str = ""
    handle: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # Observed dropdown choices, each {"index": "3:1", "label": ..., "value": ...}.
    # Carried so SELECT can resolve an option without re-reading the page.
    options: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Step:
    """One cycle, kept for the history the next decision reads."""

    n: int
    operation: str
    target: Optional[str]
    label: str
    confidence: float
    latency_ms: int
    executed: bool
    detail: str = ""
    page_changed: Optional[bool] = None
    failed_open: bool = False


@dataclass
class Run:
    """The whole attempt: what it did, and why it stopped."""

    goal: str
    steps: List[Step] = field(default_factory=list)
    stopped: str = ""
    error: str = ""

    @property
    def solved(self) -> bool:
        return self.stopped == "done"

    def summary(self) -> Dict[str, Any]:
        decisions = [s for s in self.steps if s.executed]
        latencies = sorted(s.latency_ms for s in self.steps) or [0]
        return {
            "goal": self.goal,
            "stopped": self.stopped,
            "solved": self.solved,
            "steps": len(self.steps),
            "executed": len(decisions),
            "median_decision_ms": latencies[len(latencies) // 2],
            "total_decision_ms": sum(s.latency_ms for s in self.steps),
            "failed_open": sum(1 for s in self.steps if s.failed_open),
            "trace": [
                {"n": s.n, "op": s.operation, "target": s.target, "label": s.label[:60],
                 "conf": round(s.confidence, 3), "ms": s.latency_ms, "detail": s.detail[:80]}
                for s in self.steps
            ],
        }


# Operations the loop will refuse to guess about. If the model asks for one of these
# but names something the harness did not offer, the step is treated as malformed.
_HARNESS_OPERATIONS = {"CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}

# Stopping operations carry no target. A refused DONE/BLOCKED is the same proposal from the
# same state, so it is withdrawn by operation alone - unlike a target, which needs its index.
_STOP_OPERATIONS = ("DONE", "BLOCKED")

# Safety default: operations that change the world outside the page. A confirmation
# callback gets the last word on these, whatever the model decided.
RISKY_HINTS = ("delete", "remove account", "pay", "purchase", "buy", "checkout", "send",
               "submit order", "confirm transfer", "unsubscribe", "cancel subscription")


def _goal_named_option(goal: str, options: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    """The observed dropdown option the goal names, when exactly one is named.

    Grounding, not generation: the labels come from the page we observed, so nothing the
    model wrote enters the decision. An option matches when its label appears in the goal
    text, or when every word of the label appears in the goal. Only a unique match is
    trusted - with two candidates the goal is not specific enough to override the model, so
    the model's own choice stands.
    """
    haystack = (goal or "").lower()
    words = set(re.findall(r"[a-z0-9]+", haystack))
    matches = []
    for option in options or []:
        label = str(option.get("label", "") or "").strip().lower()
        if not label:
            continue
        label_words = [word for word in re.findall(r"[a-z0-9]+", label) if len(word) > 2]
        if label in haystack or (label_words and all(word in words for word in label_words)):
            matches.append(option)
    return matches[0] if len(matches) == 1 else None


class BrowserDecider:
    """Run a goal to completion with a local decision model choosing every step.

    Example:
        from localdecide import BrowserDecider
        from localdecide.drivers import PlaywrightDriver

        with PlaywrightDriver() as driver:
            run = BrowserDecider().run(driver, "Find flights Zurich to London on 2026-09-20")
            print(run.summary())
    """

    def __init__(
        self,
        decider: Optional[Decider] = None,
        *,
        max_steps: int = 30,
        text_provider: Optional[Callable[[str, "ElementRef"], Optional[str]]] = None,
        confirm: Optional[Callable[[str, "ElementRef"], bool]] = None,
        on_step: Optional[Callable[[Step], None]] = None,
        text_chars: int = 1200,
        scope: Optional[Scope] = None,
        min_confidence: float = 0.15,
        recovery: Optional[Callable[[str, Dict[str, Any], List[Dict[str, Any]]], Optional[tuple[str, str]]]] = None,
        action_guard: Optional[Callable[[str, "ElementRef", Dict[str, Any]], bool]] = None,
        success_check: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        # The decider is created lazily on first use. Constructing a BrowserDecider is
        # something you do while wiring an agent together - inspecting attributes, testing
        # guard logic - and none of that should require a checkpoint to be installed. The
        # model loads the moment a real decision is asked for, and not before.
        self._decider = decider
        self._decider_args: tuple = ()
        if decider is None:
            self._decider_args = (Decider,)
        self.max_steps = max_steps
        # TYPE_TEXT needs a string; a decision model cannot write one. Supply a callback
        # (a small LLM, a regex over the goal, a lookup table) or TYPE_TEXT is refused.
        self.text_provider = text_provider
        self.confirm = confirm
        self.on_step = on_step
        self.text_chars = text_chars
        # Scoping is the single biggest lever on both latency and accuracy: 20 elements
        # decide in ~330 ms, 120 elements take ~1.2 s and make more mistakes.
        self.scope = scope if scope is not None else Scope()
        # Measured: on an ambiguous page the checkpoint fired a submit button with 6%
        # confidence. Acting on a near-coin-flip is worse than not acting, so anything under
        # this bar is refused. Set 0.0 to disable, or raise it in high-stakes flows.
        self.min_confidence = min_confidence
        self.recovery = recovery
        self.action_guard = action_guard
        self.success_check = success_check

    @property
    def decider(self) -> Decider:
        """The decision layer, created on first touch. May be any Decider-like object."""
        if self._decider is None:
            self._decider = self._decider_args[0]()
        return self._decider

    def run(self, driver: Driver, goal: str) -> Run:
        run = Run(goal=goal)
        if not goal.strip():
            run.stopped, run.error = "error", "empty goal"
            return run
        history: List[Dict[str, Any]] = []
        # (operation, target) pairs that already failed to move things, keyed by the state
        # they were tried from. The hash comes from the driver, so this is live for drivers
        # that report one and stays inert for drivers that do not.
        tried: Dict[str, set] = {}

        def withdraw(op: str, tgt: Optional[str]) -> None:
            """Withdraw an action that made no progress from this state's next question.

            Refusals count. A guard that refused leaves the page exactly as it was, and the model
            is deterministic, so leaving the option on offer means it proposes the same refused
            action until the step budget runs out. Measured on the hard fixture: 60 identical
            "would untick an already-checked control" refusals in one run.

            Targeted operations are keyed by (operation, target). A stopping operation (DONE,
            BLOCKED) carries no target, so it is keyed by operation alone and a DONE the success
            oracle refused is withdrawn exactly like any other no-progress action.
            """
            if state_hash is None:
                return
            if tgt is not None and op in TARGETED_OPERATIONS:
                tried.setdefault(state_hash, set()).add((op, tgt))
            elif op in _STOP_OPERATIONS:
                tried.setdefault(state_hash, set()).add((op, None))

        try:
            for number in range(1, self.max_steps + 1):
                observation = driver.observe()
                if self.scope is not None:
                    # The goal goes in: it is what protects a legitimate target ("Random
                    # article" is a nav link AND the thing the user asked for) from being
                    # mistaken for page furniture.
                    observation = self.scope.apply(observation, goal=goal)
                if self.success_check and self.success_check(observation):
                    run.steps.append(Step(number, "DONE", None, "", 1.0, 0, False, detail="success oracle"))
                    self._emit(run.steps[-1])
                    run.stopped = "done"
                    return run
                table = build_element_table(observation)
                table.history = list(history)
                questions = table_to_questions(table, goal)
                # Withdraw whatever has already been tried from this exact state. The
                # checkpoint answers the same question the same way every time, so without
                # this the run repeats one action until the loop guard stops it.
                state_hash = observation.get("state_hash")
                if state_hash is not None and tried.get(state_hash):
                    questions = drop_tried_options(questions, tried[state_hash])
                decision = self.decider.decide(table.state(text_chars=self.text_chars), questions)

                if not decision.ok:
                    step = Step(number, "WAIT", None, "", 0.0, decision.latency_ms, False,
                                detail=f"failed open: {decision.error}", failed_open=True)
                    run.steps.append(step)
                    self._emit(step)
                    # A failed-open decision is not an action. Retry once, then give up.
                    if sum(1 for s in run.steps if s.failed_open) >= 2:
                        run.stopped, run.error = "error", decision.error or "decision failed twice"
                        return run
                    continue

                answers = decision.answers
                assert answers is not None
                operation = answers.choice("operation")
                confidence = answers.confidence("operation")
                element: Optional[ElementRef] = None
                target: Optional[str] = None
                recovery_applied = False

                if (
                    self.recovery
                    and (
                        (observation.get("state_hash") == observation.get("previous_state_hash"))
                        or (
                            history
                            and (
                                str(history[-1].get("detail", "")).startswith("action guard:")
                                or str(history[-1].get("detail", "")).startswith("text provider returned nothing")
                                or str(history[-1].get("detail", "")).startswith("refused:")
                                # A driver can reject a real, observed dropdown option
                                # (``option ... not present``) without mutating the page.
                                # Treat that exactly like a guard/refusal: the next
                                # unchanged observation is a no-progress cycle and the
                                # caller's recovery policy needs a chance to propose a
                                # different observed option.
                                or str(history[-1].get("detail", "")).startswith("option ")
                            )
                        )
                    )
                    and history
                ):
                    proposal = self.recovery(goal, observation, history)
                    if proposal is not None and tuple(proposal[:2]) in (tried.get(state_hash) or ()):
                        # Recovery must not re-propose what this state has already refused.
                        proposal = None
                    if proposal is not None:
                        recovery_operation, recovery_target = proposal
                        recovery_element = table.targets_for(recovery_operation).get(recovery_target)
                        if recovery_element is None:
                            run.stopped, run.error = "error", "recovery proposed unsupported target"
                            return run
                        operation = recovery_operation
                        target = recovery_target
                        element = ElementRef(
                            recovery_element.index,
                            recovery_element.label,
                            recovery_element.role,
                            recovery_element.handle,
                            {**recovery_element.meta, "checked": recovery_element.checked, "options": recovery_element.options},
                            options=list(recovery_element.options),
                        )
                        confidence = 1.0
                        recovery_applied = True

                if not recovery_applied and operation in ("CLICK", "TYPE_TEXT", "SELECT"):
                    question_name = f"{operation.lower()}_target"
                    if question_name in answers.raw:
                        target = answers.choice(question_name)
                        found = table.targets_for(operation).get(target)
                        if found is None:
                            step = Step(number, operation, target, "", confidence, decision.latency_ms, False,
                                        detail="model named an index that does not support this operation")
                            run.steps.append(step)
                            self._emit(step)
                            run.stopped, run.error = "error", f"hallucinated target {target}"
                            return run
                        element = ElementRef(found.index, found.label, found.role, found.handle,
                                             {**found.meta, "checked": found.checked,
                                              "options": found.options},
                                             options=list(found.options))
                        confidence = min(confidence, answers.confidence(question_name))
                    else:
                        step = Step(number, operation, None, "", confidence, decision.latency_ms, False,
                                    detail="operation needs a target this page never offered")
                        run.steps.append(step)
                        self._emit(step)
                        run.stopped, run.error = "error", "no target question"
                        return run

                if operation == "DONE":
                    # The model's DONE is advisory; the success oracle owns the verdict. A
                    # success_check is already consulted at the top of each cycle, so if one is
                    # configured and the run is still here, the oracle has just disagreed. Treat
                    # the DONE as a no-progress action - refuse it, withdraw it for this state,
                    # and keep going - instead of stopping on the model's say-so while the page
                    # is not green.
                    if self.success_check is not None and not self.success_check(observation):
                        step = Step(number, "DONE", None, "", confidence, decision.latency_ms, False,
                                    detail="refused: success oracle disagrees")
                        run.steps.append(step)
                        self._emit(step)
                        history.append({"action": operation, "kind": "done", "target": None,
                                        "text": None, "page_changed": False,
                                        "detail": "refused: success oracle disagrees"})
                        withdraw("DONE", None)
                        continue
                    run.steps.append(Step(number, "DONE", None, "", confidence, decision.latency_ms, False))
                    self._emit(run.steps[-1])
                    run.stopped = "done"
                    return run
                if operation == "BLOCKED":
                    run.steps.append(Step(number, "BLOCKED", None, "", confidence, decision.latency_ms, False))
                    self._emit(run.steps[-1])
                    run.stopped = "blocked"
                    return run
                if operation not in _HARNESS_OPERATIONS:
                    run.steps.append(Step(number, operation, target, "", confidence, decision.latency_ms, False,
                                          detail="unsupported operation"))
                    self._emit(run.steps[-1])
                    run.stopped, run.error = "error", f"unsupported operation {operation!r}"
                    return run

                # Confidence gate: a decision the model is unsure about is not an action.
                # Measured: on an ambiguous page the checkpoint proposed CLICK on a submit
                # button with p=0.06. Executing that is worse than asking again, and worse
                # still than letting the caller take over, so low-confidence steps are
                # refused and recorded. DONE/BLOCKED are exempt: stopping is always safe.
                if operation not in ("DONE", "BLOCKED") and confidence < self.min_confidence:
                    run.steps.append(Step(number, operation, target, element.label if element else "",
                                          confidence, decision.latency_ms, False,
                                          detail=f"refused: confidence {confidence:.2f} below {self.min_confidence:.2f}"))
                    self._emit(run.steps[-1])
                    history.append({"action": operation, "kind": operation.lower(), "target": target,
                                    "text": None, "page_changed": False})
                    low = sum(1 for s in run.steps if "below" in s.detail)
                    if low >= 2:
                        run.stopped, run.error = "error", (
                            f"model is not confident enough to act (last: {confidence:.2f})")
                        return run
                    withdraw(operation, target)
                    continue

                if (
                    self.action_guard
                    and operation in ("CLICK", "TYPE_TEXT", "SELECT")
                    and element is not None
                    and not self.action_guard(goal, element, observation)
                ):
                    step = Step(number, operation, target, element.label, confidence,
                                decision.latency_ms, False, detail="action guard: prerequisites unmet")
                    run.steps.append(step)
                    self._emit(step)
                    history.append({"action": operation, "kind": operation.lower(), "target": target,
                                    "target_label": element.label, "text": None,
                                    "page_changed": False, "detail": "action guard: prerequisites unmet"})
                    withdraw(operation, target)
                    continue

                # Human gate: irreversible-looking actions stop here unless the caller
                # has supplied a confirmation callback that says yes. One case is not a
                # confirmation question at all - a control that destroys state the goal never
                # asked for. Measured on the multifield fixture: once the form was complete the
                # model clicked "Reset progress", the page cleared every field, and the run
                # ended with nothing filled. No "yes" makes that the right action, so it is
                # refused and taken off this state's table instead of offered to `confirm`.
                if operation in ("CLICK", "TYPE_TEXT", "SELECT") and (
                    self._looks_risky(element, goal)
                    or (operation == "CLICK" and self._destructive_unasked(element, goal))
                ):
                    if operation == "CLICK" and self._destructive_unasked(element, goal):
                        step = Step(number, operation, target, element.label if element else "",
                                    confidence, decision.latency_ms, False,
                                    detail="refused: destructive control the goal does not ask for")
                        run.steps.append(step)
                        self._emit(step)
                        history.append({"action": operation, "kind": operation.lower(), "target": target,
                                        "target_label": element.label if element else "", "text": None,
                                        "page_changed": False,
                                        "detail": "refused: destructive control the goal does not ask for"})
                        withdraw(operation, target)
                        continue
                    if self.confirm is None or not self.confirm(f"{operation} {element.label if element else ''}", element):  # type: ignore[arg-type]
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="needs confirmation"))
                        self._emit(run.steps[-1])
                        run.stopped = "needs_confirmation"
                        return run

                # Toggle guard: measured on the real checkpoint, a small decision model will
                # confidently click a checkbox that is ALREADY in the requested state (p=0.90
                # on `checked=true` with "tick the terms box" as the goal), even though the
                # option text says `checked=true` and the instructions say not to re-toggle.
                # That click would silently UNDO the user's intention, so the harness refuses
                # it and asks for a fresh decision instead. Same idea as the loop guard: the
                # model proposes, the harness checks what the proposal would actually do.
                if operation == "CLICK" and element is not None and element.meta.get("checked") is True:
                    if self._goal_wants_unticked(goal) is False:  # goal wants it ticked; it is
                        run.steps.append(Step(number, operation, target, element.label,
                                              confidence, decision.latency_ms, False,
                                              detail="refused: would untick an already-checked control"))
                        self._emit(run.steps[-1])
                        history.append({"action": operation, "kind": "click", "target": target,
                                        "target_label": element.label, "text": None,
                                        "page_changed": False,
                                        "detail": "refused: would untick an already-checked control"})
                        withdraw(operation, target)
                        continue

                text: Optional[str] = None
                if operation == "TYPE_TEXT":
                    if self.text_provider is None:
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="no text provider"))
                        self._emit(run.steps[-1])
                        run.stopped, run.error = "error", "TYPE_TEXT without a text provider"
                        return run
                    text = self.text_provider(goal, element)  # type: ignore[arg-type]
                    if not text:
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="text provider returned nothing; action skipped"))
                        self._emit(run.steps[-1])
                        history.append({"action": operation, "kind": operation.lower(), "target": target,
                                        "target_label": element.label if element else "", "text": None,
                                        "page_changed": False, "detail": "text provider returned nothing; action skipped"})
                        withdraw(operation, target)
                        continue

                # A dropdown is two answers: which field, and which option inside it. The
                # option matters, so fetch it here rather than letting the driver guess. When
                # the goal names one observed option, that grounding wins: the label comes
                # from the page, so the model's index is not trusted for it.
                option: Optional[str] = None
                option_key: Optional[str] = None
                if operation == "SELECT":
                    observed = list((element.meta.get("options") if element else None) or [])
                    option_question = (f"select_option_{target}"
                                       if f"select_option_{target}" in questions else "select_option")
                    named = _goal_named_option(goal, observed)
                    if named is not None:
                        matched = named
                    elif option_question not in answers.raw:
                        # No observed options for this dropdown, so nothing can be selected.
                        # Withdraw it and keep going: an empty dropdown is not fatal.
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False,
                                              detail="SELECT on a dropdown with no observed options"))
                        self._emit(run.steps[-1])
                        history.append({"action": operation, "kind": "select", "target": target,
                                        "target_label": element.label if element else "", "text": None,
                                        "page_changed": False,
                                        "detail": "SELECT on a dropdown with no observed options"})
                        withdraw(operation, target)
                        continue
                    else:
                        option_key = answers.choice(option_question)
                        matched = None
                        for candidate in observed:
                            if str(candidate.get("index")) == str(option_key):
                                matched = candidate
                                break
                        if matched is None:
                            run.steps.append(Step(number, operation, target, element.label if element else "",
                                                  confidence, decision.latency_ms, False,
                                                  detail=f"option {option_key!r} was not in the observed dropdown"))
                            self._emit(run.steps[-1])
                            run.stopped, run.error = "error", "hallucinated dropdown option"
                            return run
                    option = str(matched.get("value") or matched.get("label") or "")
                    if not option:
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False,
                                              detail="observed dropdown option has no value"))
                        self._emit(run.steps[-1])
                        run.stopped, run.error = "error", "empty dropdown option"
                        return run
                    if option_question in answers.raw:
                        confidence = min(confidence, answers.confidence(option_question))

                # Loop guard: same operation on the same target twice with no page change.
                repeats = sum(1 for item in history[-2:]
                              if item.get("action") == operation and item.get("target") == target
                              and item.get("page_changed") is False)
                if repeats >= 2:
                    run.steps.append(Step(number, operation, target, element.label if element else "",
                                          confidence, decision.latency_ms, False, detail="loop guard: no page change"))
                    self._emit(run.steps[-1])
                    run.stopped, run.error = "error", "stuck: repeated action with no page change"
                    return run

                try:
                    # `text` doubles as the payload for SELECT (the option value): one
                    # param, because both are "the string this operation needs".
                    payload = text if operation != "SELECT" else option
                    result = driver.execute(operation, element, payload) or {}
                except Exception as error:  # a driver failure is not the model's fault
                    step = Step(number, operation, target, element.label if element else "",
                                confidence, decision.latency_ms, False, detail=f"driver error: {error}")
                    run.steps.append(step)
                    self._emit(step)
                    run.stopped, run.error = "error", f"driver error: {error}"
                    return run

                step = Step(number, operation, target, element.label if element else "", confidence,
                            decision.latency_ms, bool(result.get("ok", True)),
                            detail=str(result.get("detail", "")),
                            page_changed=result.get("page_changed"))
                run.steps.append(step)
                self._emit(step)
                history.append({"action": operation, "kind": operation.lower(), "target": target,
                                "target_label": element.label if element else "", "text": text,
                                "page_changed": result.get("page_changed"),
                                # Preserve driver failures for the next cycle. In particular,
                                # a non-mutating SELECT rejection must be visible to the
                                # recovery policy instead of looking like a fresh action.
                                "detail": str(result.get("detail", ""))})
                # Remember only what made no progress. Something that moved the page is still
                # worth offering if the page returns here; something that did not can only
                # repeat itself, so it is withdrawn from this state's next question.
                # ponytail: a cycle whose every edge changes the state is not covered by this;
                # recovery and max_steps still own that case.
                if not result.get("ok", True) or result.get("page_changed") is False:
                    withdraw(operation, target)
            run.stopped = "max_steps"
            return run
        finally:
            try:
                driver.close()
            except Exception:
                pass

    def _looks_risky(self, element: Optional[ElementRef], goal: str) -> bool:
        haystack = f"{element.label if element else ''} {goal}".lower()
        return any(hint in haystack for hint in RISKY_HINTS)

    # Controls that destroy page state. A goal may plainly ask for one ("delete my account"),
    # in which case the hint word is in the goal text and the control is left to the caller's
    # guards. When the goal never mentions it, running it can only undo the run's own work.
    _DESTRUCTIVE_HINTS = ("reset", "expire", "start over", "discard", "wipe", "erase",
                          "abort", "delete", "remove account", "unsubscribe",
                          "clear form", "clear all", "clear progress")

    def _destructive_unasked(self, element: Optional[ElementRef], goal: str) -> bool:
        """True when the control destroys state and the goal never asks for it."""
        label = (element.label if element else "").lower()
        lowered = (goal or "").lower()
        return any(hint in label and hint not in lowered for hint in self._DESTRUCTIVE_HINTS)

    # Words that mean "this control should end up checked". Anything else (uncheck, clear,
    # opt out, deselect, disable) means the opposite, and asked-for-unticking is honoured.
    _WANT_TICKED = ("tick", "check", "enable", "opt in", "turn on", "select the", "agree", "accept")
    _WANT_UNTICKED = ("untick", "uncheck", "unselect", "deselect", "disable", "opt out",
                      "turn off", "clear the", "remove the check", "disagree")

    def _goal_wants_unticked(self, goal: str) -> Optional[bool]:
        """What does the goal want done to a checkbox: False = check it, True = uncheck it.

        Returns None when the goal does not say, in which case the toggle guard stands down -
        refusing an action on an ambiguous goal would be worse than letting it through.
        """
        lowered = (goal or "").lower()
        for word in self._WANT_UNTICKED:
            if word in lowered:
                return True
        for word in self._WANT_TICKED:
            if word in lowered:
                return False
        return None

    def _emit(self, step: Step) -> None:
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:
                pass
