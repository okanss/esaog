"""Simulated tool runtime exposed through an MCP-style JSON-RPC tool server, plus
runtime adapters (LangGraph StateGraph and a direct sequential adapter).

The runtime is the only component that sees ground truth (via `oracle`). Transient
failures use common random numbers keyed by (seed, instance, slot, actor, k-th call)
so that the same decision faces the same luck under every method.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import oracle
from .simllm import krng


@dataclass
class Attempt:
    slot: str
    actor: str
    status: str
    failure: str | None
    violations: list
    correct_assignment: bool
    cost: float
    latency: float
    artifact: dict | None = None


class SimRuntime:
    def __init__(self, inst, seed):
        self.inst, self.seed, self.dom = inst, seed, inst["domain"]
        self.actors = {a["uri"]: a for a in inst["available_actors"]}
        self.present = set(self.actors)
        self.rel = inst["oracle"]["true_reliability"]
        self.gold = {t["slot"]: t for t in inst["required_tasks"]}
        self.events = {e["slot"]: e for e in inst["perturbation"]["runtime_events"]}
        self.slot_attempts, self.calls_per = {}, {}
        self.attempts: list[Attempt] = []
        self.failed_by_event = {}  # slot -> actor removed by an injected event
        self.art_ctr = 0

    def invoke(self, slot, actor_uri, inputs: dict) -> Attempt:
        """inputs: {source slot or 'goal': artifact dict(type, correct, uri)}"""
        n = self.slot_attempts[slot] = self.slot_attempts.get(slot, 0) + 1
        gold = self.gold.get(slot)
        a = self.actors.get(actor_uri)
        in_types = [x["type"] for x in inputs.values()]
        ev = self.events.get(slot)
        if ev and n == ev["attempt"] and a is not None:
            self.present.discard(actor_uri)
            self.failed_by_event[slot] = actor_uri
            return self._log(Attempt(slot, actor_uri, "failure", ev["failure"], [], False, 0.0, 0.5))
        if a is None or actor_uri not in self.present:
            return self._log(Attempt(slot, actor_uri, "failure", "AgentUnavailable", [], False, 0.0, 0.5))
        viol = []
        if not oracle.cap_ok(self.dom, a, gold["required_capability"]):
            viol.append("CapabilityNotEntailed")
        if not oracle.output_ok(self.dom, a, gold["out_type"]):
            viol.append("IOIncompatible")
        viol += oracle.policy_violations(self.dom, a, gold)
        needed = [i for i in gold["inputs"]]
        if any(i not in inputs for i in needed):
            return self._log(Attempt(slot, actor_uri, "failure", "MissingInput", viol, False, 0.0, 0.3))
        if not oracle.input_ok(self.dom, a, in_types):
            viol = ["IOIncompatible"] + [v for v in viol if v != "IOIncompatible"]
            return self._log(Attempt(slot, actor_uri, "failure", "TypeMismatch", viol, False, a["cost"] * 0.2, 0.5))
        k = self.calls_per[(slot, actor_uri)] = self.calls_per.get((slot, actor_uri), 0) + 1
        ok_assign = not viol
        if krng(self.seed, self.inst["instance_id"], "exec", slot, actor_uri, k).random() > self.rel[actor_uri]:
            return self._log(Attempt(slot, actor_uri, "failure", "TransientError", viol, ok_assign, a["cost"], a["latency"]))
        self.art_ctr += 1
        correct = (not any(v in ("CapabilityNotEntailed", "IOIncompatible") for v in viol)
                   and all(x["correct"] for x in inputs.values()))
        art = dict(uri=f"{self.inst['goal']['uri']}/artifact/{slot}/{self.art_ctr}", type=a["produces"], correct=correct,
                   slot=slot)
        return self._log(Attempt(slot, actor_uri, "success", None, viol, ok_assign, a["cost"], a["latency"], art))

    def _log(self, at):
        self.attempts.append(at)
        return at

    def goal_artifact(self):
        g = self.inst["goal"]
        return dict(uri=g["uri"] + "/input", type=g["provides_input"], correct=True, slot="goal")


class MCPToolServer:
    """Minimal MCP-compatible surface (JSON-RPC 2.0 `tools/list` and `tools/call`)."""

    def __init__(self, runtime: SimRuntime):
        self.rt = runtime
        self.by_tool = {a["mcp_tool"]: a["uri"] for a in runtime.inst["available_actors"]}

    def handle(self, request: str) -> str:
        req = json.loads(request)
        if req["method"] == "tools/list":
            tools = [dict(name=a["mcp_tool"], description=a["description"],
                          inputSchema={"type": "object", "properties": {"slot": {"type": "string"},
                                                                        "inputs": {"type": "object"}},
                                       "required": ["slot", "inputs"]})
                     for a in self.rt.inst["available_actors"] if a["uri"] in self.rt.present]
            return json.dumps(dict(jsonrpc="2.0", id=req["id"], result=dict(tools=tools)))
        if req["method"] == "tools/call":
            p = req["params"]
            actor = self.by_tool.get(p["name"], p["name"])
            at = self.rt.invoke(p["arguments"]["slot"], actor, p["arguments"]["inputs"])
            body = dict(status=at.status, failure=at.failure, artifact=at.artifact)
            return json.dumps(dict(jsonrpc="2.0", id=req["id"],
                                   result=dict(content=[dict(type="text", text=json.dumps(body))],
                                               isError=at.status != "success")))
        return json.dumps(dict(jsonrpc="2.0", id=req.get("id"), error=dict(code=-32601, message="method not found")))


class MCPClient:
    def __init__(self, server: MCPToolServer):
        self.server, self._id = server, 0
        self.tool_of = {v: k for k, v in server.by_tool.items()}

    def call(self, actor_uri, slot, inputs) -> dict:
        self._id += 1
        req = dict(jsonrpc="2.0", id=self._id, method="tools/call",
                   params=dict(name=self.tool_of.get(actor_uri, actor_uri), arguments=dict(slot=slot, inputs=inputs)))
        resp = json.loads(self.server.handle(json.dumps(req)))
        return json.loads(resp["result"]["content"][0]["text"])


# ---------------------------------------------------------------------- adapters
class DirectAdapter:
    """Runs committed steps sequentially; stops at the first failed step."""
    name = "direct"

    def run(self, steps, execute_step):
        for s in steps:
            if not execute_step(s):
                return s
        return None


class LangGraphAdapter:
    """Compiles a committed ESAOG workflow version into a LangGraph StateGraph.
    Node ids are local step names; `uri_of` preserves the mapping back to semantic URIs."""
    name = "langgraph"

    def run(self, steps, execute_step):
        from typing import TypedDict
        from langgraph.graph import StateGraph, END

        class S(TypedDict, total=False):
            failed: str

        g = StateGraph(S)
        names = [f"n{i}_{s['slot']}" for i, s in enumerate(steps)]
        self.uri_of = {n: s.get("step_uri") for n, s in zip(names, steps)}

        def mk(step):
            def node(state):
                return {"failed": ""} if execute_step(step) else {"failed": step["slot"]}
            return node

        for n, s in zip(names, steps):
            g.add_node(n, mk(s))
        g.set_entry_point(names[0])
        for i, n in enumerate(names):
            nxt = names[i + 1] if i + 1 < len(names) else END
            g.add_conditional_edges(n, lambda st, nxt=nxt: END if st.get("failed") else nxt)
        out = g.compile().invoke({"failed": ""}) or {}
        failed = out.get("failed")
        return next((s for s in steps if s["slot"] == failed), None) if failed else None
