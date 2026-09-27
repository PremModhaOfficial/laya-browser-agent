"""MCP server: expose the decision model to Claude Desktop, Cursor, and any MCP client.

Why this matters: Model Context Protocol is how agent stacks consume external tools now.
Serving the model over MCP means a Claude Desktop config block or a Cursor setting is the
whole integration - no code, no SDK, no fork. The same local model, the same guarantees:
nothing leaves the machine.

Two tools are exposed:

* `decide`     - typed questions about a state (the primitive)
* `page_decide` - an observation and a goal, get the chosen operation and element

Run it:

    localdecide-mcp                # stdio transport, the standard for desktop clients

Register it (Claude Desktop, `claude_desktop_config.json`):

    { "mcpServers": { "localdecide": { "command": "/path/to/localdecide-mcp" } } }

The protocol is implemented with nothing but stdio and JSON - no MCP SDK dependency - so
this works on any Python the package already supports.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from .browser_tools import TOOL_NAMES as BROWSER_TOOL_NAMES
from .browser_tools import BrowserTools
from .decider import Decider
from .page import build_element_table, table_to_questions
from .registry import RegistryFull

# Protocol versions this server speaks, newest first. Per the MCP lifecycle
# spec (2025-06-18 §Version Negotiation): if the client requests a version we
# support, respond with the same one; otherwise respond with our latest.
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2024-11-05"]
PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]


def _package_version() -> str:
    """Same single source as localdecide.__version__ — metadata, not a constant."""
    from . import __version__
    return __version__


SERVER_INFO = {"name": "localdecide", "version": _package_version()}

TOOLS = [
    {
        "name": "decide",
        "description": (
            "Ask a local, open-weight System 1 decision model typed questions about a state. "
            "Returns calibrated probabilities, never generated text. Use for classification, "
            "routing, scoring, and yes/no judgments where the answer is one of a few options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "description": "The state to judge: a string, or a JSON object of facts.",
                    "type": ["string", "object"],
                },
                "questions": {
                    "description": (
                        "Questions keyed by name. Each: {type: choice|score|noul, instructions, "
                        "criteria}. choice takes a dict of option->description; score takes an "
                        "ordered list of levels; noul takes none."
                    ),
                    "type": "object",
                },
            },
            "required": ["state", "questions"],
        },
    },
    {
        "name": "page_decide",
        "description": (
            "Decide the next browser action: give a page observation (url, title, text, and an "
            "actions/elements list from a DOM or accessibility snapshot) plus a goal, and get "
            "the chosen operation (CLICK/TYPE_TEXT/SELECT/SCROLL/WAIT/DONE/BLOCKED) and the "
            "index of the element to act on. All local; the page never leaves this machine."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"description": "What the agent is trying to do.", "type": "string"},
                "observation": {
                    "description": (
                        "Page observation: {url, title, text, actions: [{kind: click|fill|select, "
                        "label, role, node?}]} or {elements: [...]}."
                    ),
                    "type": "object",
                },
            },
            "required": ["goal", "observation"],
        },
    },
]


class _Session:
    """One stdio MCP session. Loads the model lazily, answers one request at a time."""

    def __init__(self, browser_tools: "BrowserTools | None" = None) -> None:
        self._decider: Decider | None = None
        # The seven browser tools, when the server was given a surface. They are optional so
        # the two decision tools keep working in a deployment that has no browser, and so the
        # model-free check can drive the surface without a browser at all.
        self._browser = browser_tools

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Every tool this server exposes: the decision tools, plus the browser seven if wired."""
        if self._browser is None:
            return list(TOOLS)
        return list(TOOLS) + self._browser.list_tools()

    def decider(self) -> Decider:
        if self._decider is None:
            import os

            self._decider = Decider(max_options_per_question=int(os.environ.get("LOCALDECIDE_MAX_OPTIONS", "20")))
        return self._decider

    # -- handlers ----------------------------------------------------------

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        method = message.get("method", "")
        request_id = message.get("id")
        try:
            if method == "initialize":
                requested = str(message.get("params", {}).get("protocolVersion", "") or "")
                # Spec: same version if we support it, else our latest.
                version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
                return self._ok(request_id, {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                })
            if method == "notifications/initialized":
                return {}  # notification: no response
            if method == "tools/list":
                return self._ok(request_id, {"tools": self.tools})
            if method == "tools/call":
                return self._ok(request_id, self._call_tool(message.get("params", {})))
            if method == "ping":
                return self._ok(request_id, {})
            return self._error(request_id, -32601, f"method not found: {method}")
        except Exception as error:
            return self._error(request_id, -32603, f"{type(error).__name__}: {error}")

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        # The browser seven are dispatched first, so a name collision can never shadow a
        # decision tool: the two sets are disjoint by construction (start/run/step/observe/
        # answer/state/stop vs decide/page_decide).
        if name in BROWSER_TOOL_NAMES:
            return self._call_browser(name, arguments)
        if name == "decide":
            decision = self.decider().decide(arguments.get("state", ""), arguments.get("questions") or {})
            if not decision.ok:
                return self._tool_error(f"decision failed open: {decision.error}")
            assert decision.answers is not None
            return {"content": [{"type": "text", "text": json.dumps(
                {"answers": decision.answers.raw, "latency_ms": decision.latency_ms,
                 "backend": decision.answers.backend}, ensure_ascii=False, default=str)}]}
        if name == "page_decide":
            goal = str(arguments.get("goal", "") or "")
            observation = arguments.get("observation") or {}
            if not goal.strip():
                return self._tool_error("goal is required")
            table = build_element_table(observation)
            questions = table_to_questions(table, goal)
            decision = self.decider().decide(table.state(text_chars=1200), questions)
            if not decision.ok:
                return self._tool_error(f"decision failed open: {decision.error}")
            assert decision.answers is not None
            answers = decision.answers
            operation = answers.choice("operation")
            out: Dict[str, Any] = {"operation": operation,
                                   "confidence": answers.confidence("operation"),
                                   "latency_ms": decision.latency_ms, "backend": answers.backend}
            if operation in ("CLICK", "TYPE_TEXT", "SELECT") and f"{operation.lower()}_target" in answers.raw:
                target = answers.choice(f"{operation.lower()}_target")
                element = table.by_index().get(target)
                out.update(target=target, label=(element.label if element else ""),
                           handle=(element.handle if element else None))
                out["confidence"] = min(out["confidence"], answers.confidence(f"{operation.lower()}_target"))
            return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=str)}]}
        return self._tool_error(f"unknown tool: {name!r}")

    def _call_browser(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one of the browser seven through the surface.

        A refused call (the cap, an unknown run, a wrong-field answer) comes back as a tool
        error carrying the reason, not as a transport exception: a client must be able to tell
        "busy" from "broken", and an opaque -32603 destroys that distinction.
        """
        if self._browser is None:
            return self._tool_error(f"{name}: this server was started without the browser tools")
        try:
            payload = self._browser.call(name, arguments)
        except KeyError as error:            # unknown run id
            return self._tool_error(f"{name}: {error}")
        except RegistryFull as error:        # the cap, surfaced as a refusal
            return self._tool_error(f"{name}: registry full ({error.active}/{error.cap}); refused")
        if isinstance(payload, dict) and payload.get("error"):
            text = str(payload["error"])
            if payload.get("refused"):
                text += " (refused, not queued)"
            return self._tool_error(f"{name}: {text}")
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}]}

    @staticmethod
    def _tool_error(message: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- framing -----------------------------------------------------------

    @staticmethod
    def _ok(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def build_browser_tools() -> BrowserTools:
    """A live browser surface: a real Playwright driver per run, and the real session factory.

    Constructed here rather than at module import so that importing this file for its decision
    tools opens no browser and loads no checkpoint. One driver per run is also what makes a
    resume meaningful: the run keeps the page it was on.
    """
    from .drivers import PlaywrightDriver
    from .loop import BrowserDecider
    from .session import Session

    def factory(run_id: str, spec: Dict[str, Any]) -> Session:
        # The spec's url is the starting page. A client that omits it gets about:blank, which
        # the driver's own default also says - made explicit here so the spec is the single read.
        return Session(loop=BrowserDecider(),
                       driver=PlaywrightDriver(start_url=spec.get("url") or "about:blank"),
                       goal="")

    return BrowserTools(factory)


def serve(with_browser: bool = True) -> None:
    """Run the MCP server over stdio until stdin closes.

    With `with_browser` the seven browser tools are exposed alongside the two decision tools,
    which is what a client calling `tools/list` needs to see. The surface is built here rather
    than at import so that `with_browser=False` stays a pure decision server, and so a failure
    to construct a browser surface cannot stop the decision tools from serving.
    """
    browser: BrowserTools | None = None
    if with_browser:
        try:
            browser = build_browser_tools()
        except Exception:  # a browser that will not open must not blind the decision tools
            browser = None
    session = _Session(browser)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = session.handle(message)
        if response:  # notifications produce no response
            sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve()
