"""The seven browser tools, as a stateless MCP surface (plan step 7).

`mcp_server.py` already speaks JSON-RPC over stdio and exposes two *decision* tools
(`decide`, `page_decide`). This module is the *browser* half: `start`, `run`, `step`,
`observe`, `answer`, `state`, `stop`, exactly the seven the plan names, layered on the
`Session` (one engine, `session.py`) and the `RunRegistry` (cap, per-run lock, sliding
expiry, `registry.py`).

Stateless at the protocol level, stateful in the registry. "Stateless" here means the same
as the 2026-07-28 spec intends: the *protocol* carries no session token and every tool call
is self-describing - a client may reconnect, replay, or interleave calls and the server keeps
no per-connection conversation. The run state lives in the registry keyed by a run id, which
is the one piece of state a multi-step task cannot avoid, and it lives there rather than on
the connection precisely so a dropped connection does not lose a run. That is the whole design
decision, and it is why `state`/`answer` need a `run_id`: it is the only handle, and it is
data, not a cookie.

`run` is the default and `step` is the debugger, and both go through the same Session, so a
run driven either way produces the same actions - the invariant `check_engine_parity` pins at
the engine level and this layer must not break.

No model is loaded here. The tools take a `loop_factory` so a test can drive the whole
protocol with a scripted decider; the real server supplies one that builds a BrowserDecider
with the vendored checkpoint.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from .registry import MAX_CONCURRENT_RUNS, RegistryFull, RunRegistry
from .session import Session, UnknownField

#: The seven tools, in the order a client is likely to want them.
TOOL_NAMES = ("start", "run", "step", "observe", "answer", "state", "stop")


def _tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
    }


RUN_ID = {"type": "string", "description": "The run id returned by start/run. The only handle to a run."}
GOAL = {"type": "string", "description": "What the agent is trying to do, in the user's words."}

TOOLS = [
    _tool("start", "Launch a browser run and return its run id immediately, without driving it. "
                   "Use this to reserve a slot, then poll state, or call run. Respects the "
                   "four-run cap: a fifth concurrent start is refused, not queued.",
          {"goal": GOAL}, ["goal"]),
    _tool("run", "Drive a run to completion. This is the default verb. Returns when the run "
                 "finishes, is stopped, or is waiting on a value it cannot know (check "
                 "`waiting_for` and answer it).",
          {"goal": GOAL, "run_id": {"type": "string", "description": "Continue an existing run instead of starting one."}},
          ["goal"]),
    _tool("step", "Advance a run by exactly one turn. The debugger verb: use it to watch a run "
                  "move. Produces the same actions as run, one turn at a time.",
          {"run_id": RUN_ID, "goal": {"type": "string", "description": "Required only to start a fresh run."}},
          []),
    _tool("observe", "Read the current page without deciding anything. No inference, no state change.",
          {"run_id": RUN_ID}, ["run_id"]),
    _tool("answer", "Supply the value for the field a run is waiting on, or skip it when leaving "
                    "it empty is correct. An answer for any other field is refused.",
          {"run_id": RUN_ID, "field": {"type": "string", "description": "The field label, exactly as reported by state."},
           "value": {"type": "string", "description": "The value to type. Omit with skip=true."},
           "skip": {"type": "boolean", "description": "Decline to supply a value for this field."}},
          ["run_id", "field"]),
    _tool("state", "Where a run is: its status, steps so far, the last action, and anything it is "
                   "waiting on. Safe to poll; polling keeps a finished run alive.",
          {"run_id": RUN_ID}, ["run_id"]),
    _tool("stop", "Stop a run and release its browser. Idempotent.",
          {"run_id": RUN_ID}, ["run_id"]),
]


class BrowserTools:
    """The tool surface over a registry and a session factory. No JSON-RPC here.

    Kept transport-free on purpose: `mcp_server.py` wraps this, and the model-free check
    drives this directly. The two cannot drift because there is only one implementation.
    """

    def __init__(self, session_factory: Callable[[str], Session],
                 registry: Optional[RunRegistry] = None) -> None:
        # The factory takes the run id so a session can be recovered after a reconnect; that
        # is the whole point of a stateless protocol layer.
        self._factory = session_factory
        self.registry = registry if registry is not None else RunRegistry()
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(TOOLS)

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        arguments = arguments or {}
        if name not in TOOL_NAMES:
            raise KeyError(f"unknown tool: {name}")
        return getattr(self, f"_{name}")(arguments)

    # -- tools -------------------------------------------------------------

    def _start(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(arguments.get("goal", "") or "").strip()
        if not goal:
            return self._error("goal is required")
        try:
            run = self.registry.create(goal)
        except RegistryFull as full:
            # Refused, not queued: the cap is a hard resource limit, and a client that got a
            # "wait your turn" it could not honour would hang instead of failing clearly.
            return self._error(f"run registry is full ({full.active}/{full.cap} active); "
                               "refusing, not queueing", refused=True)
        self.registry.start(run.id)
        with self._lock:
            session = self._factory(run.id)
            # Park it in the run record immediately, not lazily on first use: a client can
            # reconnect between start and the first turn, and a resume that rebuilt a new
            # driver would hand back a blank page and throw the run away.
            run.data["session"] = session
            self._sessions[run.id] = session
        return {"run_id": run.id, "status": "started", "goal": goal}

    def _run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(arguments.get("goal", "") or "").strip()
        run_id = str(arguments.get("run_id", "") or "")
        if run_id:
            session = self._session(run_id)
        else:
            if not goal:
                return self._error("goal is required")
            started = self._start({"goal": goal})
            if "error" in started:
                return started
            run_id = started["run_id"]
            session = self._session(run_id)
        if goal and not session.goal:
            session.goal = goal
        session.run()
        return self._state_payload(session, {"run_id": run_id})

    def _step(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(arguments.get("run_id", "") or "")
        goal = str(arguments.get("goal", "") or "").strip()
        if run_id:
            session = self._session(run_id)
            if goal and not session.goal:
                session.goal = goal
        elif goal:
            started = self._start({"goal": goal})
            if "error" in started:
                return started
            run_id = started["run_id"]
            session = self._session(run_id)
        else:
            return self._error("run_id or goal is required")
        session.step()
        return self._state_payload(session, {"run_id": run_id})

    def _observe(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"run_id": str(arguments.get("run_id", "") or ""), "page": self._session(
            str(arguments.get("run_id", "") or "")).observe()}

    def _answer(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(arguments.get("run_id", "") or "")
        field = str(arguments.get("field", "") or "")
        session = self._session(run_id)
        try:
            outcome = session.answer(field, str(arguments.get("value", "") or ""),
                                     skip=bool(arguments.get("skip", False)))
        except UnknownField as error:
            return self._error(f"answer refused: {error}")
        return {"run_id": run_id, **outcome}

    def _state(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(arguments.get("run_id", "") or "")
        return self._state_payload(self._session(run_id), {"run_id": run_id})

    def _stop(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(arguments.get("run_id", "") or "")
        session = self._session(run_id)
        result = session.stop()
        self.registry.finish(run_id)
        return {"run_id": run_id, **result}

    # -- internals ---------------------------------------------------------

    def _session(self, run_id: str) -> Session:
        """The live session for a run, rebuilt only if this process does not hold one.

        A session is not reconstructible from a run id alone: it owns a browser and a page, and
        a fresh factory call would hand back a *new* page with the run's progress thrown away.
        So the live session is parked in the registry's run record - which is memory-only and
        dies with the process, exactly like a browser does - and this process reuses it. The
        run id remains the only handle a client needs, which is what "stateless at the protocol
        level" means: a reconnect resumes the same run, it does not start a new one.
        """
        with self._lock:
            session = self._sessions.get(run_id)
            if session is not None:
                return session
            record = self.registry.get(run_id)          # raises KeyError on an unknown run
            parked = record.data.get("session")
            if isinstance(parked, Session):
                self._sessions[run_id] = parked
                return parked
            session = self._factory(run_id)
            record.data["session"] = session
            self._sessions[run_id] = session
            return session

    def _state_payload(self, session: Session, extra: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(extra)
        payload.update(session.state())
        return payload

    @staticmethod
    def _error(message: str, refused: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {"error": message}
        if refused:
            out["refused"] = True
        return out


__all__ = ["BrowserTools", "TOOLS", "TOOL_NAMES", "MAX_CONCURRENT_RUNS"]
