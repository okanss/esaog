"""Orchestration methods: baselines B0-B8 (mechanism-equivalent re-implementations; NOT
the original published systems), the full ESAOG method P, and ablations M1-M7.

All methods share: the same SOST instance, the same SimLLM (common random numbers),
the same MCP tool server / simulated runtime, and the same attempt budget.
"""
from __future__ import annotations

import copy
import time
from collections import Counter

from rdflib import Graph, Literal, URIRef, XSD

from . import oracle
from .domains import DOMAINS
from .ontology import ESAOG as ESG, tbox
from .reasoner import asserted_closure
from .runtime import DirectAdapter, LangGraphAdapter, MCPClient, MCPToolServer, SimRuntime
from .semantic import SemanticStore, Timer, _PREP
from .simllm import Meter, SimLLM, TfIdf

MAX_ATTEMPTS = 3      # executions per task
MAX_BLOCKS = 3        # pre-invocation rejections per task (gate-based methods)
MAX_REPAIRS = 3       # planner repair rounds (A2, k)
PROV_FIELDS = ["task_uri", "step", "actor", "used", "time", "justification", "generated", "replaces"]


_PROVIDERS = {}


def get_provider(spec):
    if spec not in _PROVIDERS:
        from .providers import make_provider
        _PROVIDERS[spec] = make_provider(spec)
    return _PROVIDERS[spec]


class Orchestrator:
    name = "base"
    llm_decompose = True
    prov_fields = ("actor", "time")

    def __init__(self, inst, seed, profile="nominal"):
        self.inst, self.seed, self.dom = inst, seed, inst["domain"]
        self.ns = DOMAINS[self.dom]["ns"]
        self.rt = SimRuntime(inst, seed)
        self.mcp = MCPClient(MCPToolServer(self.rt))
        if profile.startswith("llm:"):  # real model, e.g. llm:anthropic:claude-opus-5
            from .llm_backend import RealLLM
            self.llm = RealLLM(inst, seed, get_provider(profile[4:]))
            self.meter = self.llm.meter
        else:
            self.meter = Meter()
            self.llm = SimLLM(inst, seed, profile, self.meter)
        self.timer = Timer()
        self.actors = {a["uri"]: a for a in inst["available_actors"]}
        self.gold = {t["slot"]: t for t in inst["required_tasks"]}
        self.artifacts, self.final_actor = {}, {}
        self.prov, self.notes = [], Counter()
        self.abort_reason = None
        self.tfidf = self.llm.tfidf
        self.clock = 0.0

    # ----------------------------------------------------------- planning
    def template_plan(self):
        if not self.inst["template_available"]:
            return None
        plan = []
        for t in self.inst["required_tasks"]:
            plan.append(dict(slot=t["slot"], believed=t["required_capability"], text=t["text"], task=t,
                             inputs=list(t["inputs"]), consumes=[i for i in t["inputs"] if i != "goal"],
                             goal_input=t["in_types"][0] if "goal" in t["inputs"] else None))
        return plan

    def plan(self):
        return self.llm.decompose() if self.llm_decompose else self.template_plan()

    # ----------------------------------------------------------- hooks
    def assign(self, step, excluded=()):
        raise NotImplementedError

    def gate(self, step, actor, inputs):
        return []

    def post_check(self, step, artifact):
        return True

    def on_failure(self, step, actor, failure, tried):
        nxt = self.assign(step, excluded=set(tried))
        return ("retry", nxt) if nxt else ("abort",)

    def upstream_types(self, step, inputs=None):
        if inputs is not None:
            return [x["type"] for x in inputs.values()]
        return list(step["task"]["in_types"])

    # ----------------------------------------------------------- execution
    def inputs_for(self, step):
        inp = {c: self.artifacts[c] for c in step["consumes"] if c in self.artifacts}
        if step.get("goal_input"):
            inp["goal"] = self.rt.goal_artifact()
        return inp

    def log(self, step, actor, res, justification, replaces):
        """Field-level provenance record of what this mechanism natively logs."""
        rec = {f: (f in self.prov_fields) for f in PROV_FIELDS}
        if res["status"] != "success":
            rec["generated"] = True  # not applicable
        if replaces is None:
            rec["replaces"] = True   # not a recovery decision
        self.prov.append(rec)

    def execute_step(self, step, queue, i):
        inputs = self.inputs_for(step)
        actor = step.get("agent") or self.assign(step)
        tried, blocks, attempts, prev = [], 0, 0, None
        while attempts < MAX_ATTEMPTS:
            if actor is None:
                self.abort_reason = self.abort_reason or "NoCandidate"
                return False
            v = self.gate(step, actor, inputs)
            if v:
                blocks += 1
                self.notes["blocked"] += 1
                if blocks > MAX_BLOCKS:
                    self.abort_reason = "GateExhausted"
                    return False
                tried.append(actor)
                actor = self.assign(step, excluded=set(tried), feedback=v) if self._accepts_feedback() else self.assign(step, excluded=set(tried))
                continue
            attempts += 1
            res = self.mcp.call(actor, step["slot"], inputs)
            self.clock += 1
            tried.append(actor)
            self.log(step, actor, res, f"{self.name} decision", prev)
            prev = len(self.prov)
            if res["status"] == "success":
                if self.post_check(step, res["artifact"]):
                    self.artifacts[step["slot"]] = res["artifact"]
                    self.final_actor[step["slot"]] = actor
                    return True
                self.notes["output_governance_reject"] += 1
                failure = "OutputGovernance"
            else:
                failure = res["failure"]
            if failure == "MissingInput":
                if self.insert_missing(step, queue, i):
                    return "inserted"
                self.abort_reason = "MissingInput"
                return False
            act = self.on_failure(step, actor, failure, tried)
            if act[0] == "abort":
                self.abort_reason = f"Unrecovered:{failure}"
                return False
            actor = act[1]
        self.abort_reason = "AttemptsExhausted"
        return False

    def _accepts_feedback(self):
        return False

    def insert_missing(self, step, queue, i):
        return False

    def _llm_insert(self, step, queue, i):
        """LLM replanning after a MissingInput runtime error (used by LLM-driven methods)."""
        if step.get("_inserted"):
            return False
        step["_inserted"] = True
        have = {s["slot"] for s in queue}
        missing = [x for x in self.gold[step["slot"]]["inputs"] if x != "goal" and x not in have]
        if not missing:
            return False
        new = self.llm.repair_decomposition([copy.copy(s) for s in queue], missing, key=f"rt_{step['slot']}")
        added = [s for s in new if s["slot"] in missing]
        if not added:
            return False
        for s in added:
            s["consumes"] = [x for x in self.gold[s["slot"]]["inputs"] if x != "goal"]
            s["goal_input"] = s["task"]["in_types"][0] if "goal" in s["task"]["inputs"] else None
        order = [t["slot"] for t in self.inst["required_tasks"]]
        for s in sorted(added, key=lambda s: order.index(s["slot"])):
            queue.insert(i, s)
        slots = {q["slot"] for q in queue}
        for q in queue[i:]:  # dependencies of the current and later steps now include the inserted tasks
            q["consumes"] = [x for x in self.gold[q["slot"]]["inputs"] if x != "goal" and x in slots]
        self.notes["runtime_replan"] += 1
        return True

    def run(self):
        t0 = time.perf_counter()
        plan = self.plan()
        if not plan:
            self.abort_reason = "NoPlan"
            return self.finish(time.perf_counter() - t0)
        self.pre_execute(plan)
        queue = list(plan)
        i = 0
        while i < len(queue):
            r = self.execute_step(queue[i], queue, i)
            if r == "inserted":
                continue
            if not r:
                break
            i += 1
        return self.finish(time.perf_counter() - t0)

    def pre_execute(self, plan):
        pass

    # ----------------------------------------------------------- evaluation
    def finish(self, wall):
        return evaluate(self, wall)


def evaluate(m: Orchestrator, wall):
    inst, dom = m.inst, m.dom
    atts = m.rt.attempts
    gold_slots = [t["slot"] for t in inst["required_tasks"]]
    tsr = all(s in m.artifacts and m.artifacts[s]["correct"] for s in gold_slots)
    policy_kinds = ("ClearanceViolation", "ProhibitionViolation", "BudgetViolation", "ObligationViolation")
    executed = [a for a in atts if a.status == "success" or a.failure == "TransientError"]
    pol = sum(1 for a in executed for v in a.violations if v in policy_kinds)
    io = sum(1 for a in atts if "IOIncompatible" in a.violations)
    event_fail = [a for a in atts if a.failure == "AgentUnavailable" and m.rt.failed_by_event.get(a.slot) == a.actor]
    scored = [a for a in atts if a not in event_fail]
    saa = (sum(1 for a in scored if a.correct_assignment) / len(scored)) if scored else 0.0
    cv = sum(1 for a in atts if a.violations)
    cvr = cv / len(atts) if atts else 0.0
    vws = tsr and pol == 0
    runtime_fail = any(a.status != "success" for a in atts)
    # recovery regret on runtime-perturbed slots (F, R, C)
    fam = inst["variant"]
    rel = inst["oracle"]["true_reliability"]
    rr_slots = []
    if fam in ("F", "R", "C"):
        tgt = [e["slot"] for e in inst["perturbation"]["runtime_events"]]
        if fam == "R":
            tgt = inst["perturbation"]["target_slots"]
        for s in tgt:
            removed = {m.rt.failed_by_event.get(s)}
            valid = [u for u in inst["valid_alternatives"][s] if u not in removed]
            ustar = max(oracle.utility(m.actors[u], rel[u]) for u in valid) if valid else 0.0
            fa = m.final_actor.get(s)
            ok = (fa is not None and s in m.artifacts and m.artifacts[s]["correct"] and fa in valid and vws)
            rr_slots.append(ustar - (oracle.utility(m.actors[fa], rel[fa]) if ok else 0.0))
    rr = sum(rr_slots) / len(rr_slots) if rr_slots else None
    tool_cost = sum(a.cost for a in atts)
    tool_lat = sum(a.latency for a in atts)
    sem = m.timer.t
    sem_total = sem["reasoning"] + sem["sparql"] + sem["shacl"] + sem["graph_write"]
    pc = (sum(sum(r.values()) / len(PROV_FIELDS) for r in m.prov) / len(m.prov)) if m.prov else 0.0
    pc_complete = (sum(1 for r in m.prov if all(r.values())) / len(m.prov)) if m.prov else 0.0
    # assignment-error causes
    causes = Counter()
    for a in scored:
        if a.correct_assignment:
            continue
        if a.failure == "AgentUnavailable":
            causes["unavailable_actor"] += 1
            continue
        role = inst["oracle"]["roles"].get(a.actor, "?")
        for v in a.violations:
            if v == "CapabilityNotEntailed":
                causes["capability:" + ("lexical_distractor" if "distractor" in role else role)] += 1
            else:
                causes[v] += 1
    if m.abort_reason and m.abort_reason.startswith("NoCandidate"):
        causes["false_exclusion_no_candidate"] += 1
    rec = dict(
        method=m.name, instance_id=inst["instance_id"], base_id=inst["base_id"], domain=dom, variant=fam, seed=m.seed,
        defect=bool(inst["modeling_defects"]), template=inst["template_available"],
        TSR=int(tsr), VWS=int(vws), SAA=saa, CVR=cvr, policy_violations=pol, io_incompat=io,
        runtime_failure=int(runtime_fail), recovered=int(runtime_fail and vws), RR=rr,
        PC=pc, PC_complete=pc_complete, abort=m.abort_reason or "",
        attempts=len(atts), tool_calls=len(atts), blocked=m.notes["blocked"],
        llm_calls=m.meter.calls, tokens_in=m.meter.tokens_in, tokens_out=m.meter.tokens_out,
        cost_llm=m.meter.cost, cost_tool=tool_cost, cost=m.meter.cost + tool_cost,
        lat_llm=m.meter.latency, lat_tool=tool_lat, lat_sem=sem_total, latency=m.meter.latency + tool_lat + sem_total,
        t_reasoning=sem["reasoning"], t_reasoning_cold=sem.get("reasoning_cold", 0.0), t_sparql=sem["sparql"],
        t_shacl=sem["shacl"], t_graph=sem["graph_write"], wall=wall,
        graph_triples=getattr(m, "graph_triples", 0), repairs=m.notes["repairs"], recompositions=m.notes["recompositions"],
        causes=dict(causes),
        llm_real_calls=getattr(m.meter, "real_calls", 0), llm_cache_hits=getattr(m.meter, "cache_hits", 0),
        llm_stats=dict(getattr(m.llm, "stats", {}) or {}),
        failures=dict(Counter(a.failure for a in atts if a.failure)),
    )
    return rec


# =================================================================== baselines
class B0Static(Orchestrator):
    name, llm_decompose = "B0", False
    prov_fields = ("task_uri", "step", "actor", "time", "generated")

    def plan(self):
        plan = []
        for t in self.inst["required_tasks"]:
            plan.append(dict(slot=t["slot"], believed=t["required_capability"], text=t["text"], task=t,
                             consumes=[i for i in t["inputs"] if i != "goal"],
                             goal_input=t["in_types"][0] if "goal" in t["inputs"] else None,
                             agent=self.inst["static_workflow"].get(t["slot"])))
        return plan

    def assign(self, step, excluded=()):
        return step.get("agent")

    def on_failure(self, step, actor, failure, tried):
        return ("retry", actor)  # static retry policy


class B1LLMOnly(Orchestrator):
    name = "B1"
    prov_fields = ("actor", "time", "justification")

    def present(self):
        return list(self.actors.values())

    def assign(self, step, excluded=(), feedback=None):
        ups = self.upstream_types(step, self.inputs_for(step))
        a, _ = self.llm.choose(step, self.present(), ups, info=("meta",), excluded=set(excluded),
                               key="assign", attempt=len(excluded))
        return a

    def insert_missing(self, step, queue, i):
        return self._llm_insert(step, queue, i)


class B2Embedding(Orchestrator):
    name = "B2"
    prov_fields = ("actor", "time", "justification")

    def assign(self, step, excluded=()):
        best = sorted(((self.tfidf.cos(step["text"], a["description"]), a["uri"]) for a in self.actors.values()
                       if a["uri"] not in excluded), reverse=True)
        return best[0][1] if best else None


class _GraphMixin:
    def build_asserted(self):
        t0 = time.perf_counter()
        store = SemanticStore.__new__(SemanticStore)  # reuse G_C serialisation only
        store.inst, store.domain = self.inst, self.dom
        from rdflib import Namespace
        store.ns = Namespace(self.ns)
        store.gc = Graph()
        SemanticStore._load_capabilities(store)
        self.gc = store.gc
        self.tb = tbox(self.dom, defects=[tuple(d) for d in self.inst["modeling_defects"]])
        self.asserted = self.tb + self.gc
        self.aclosure = asserted_closure(self.tb)
        self.timer.add("graph_write", time.perf_counter() - t0)

    def U(self, local):
        return URIRef(self.ns + local)

    def asub(self, c, d):
        return (self.ns + d) in self.aclosure.get(self.ns + c, {self.ns + c})


class B3GraphRouter(_GraphMixin, Orchestrator):
    name = "B3"
    prov_fields = ("task_uri", "step", "actor", "time", "justification")

    def plan(self):
        p = super().plan()
        self.build_asserted()
        return p

    def assign(self, step, excluded=()):
        t0 = time.perf_counter()
        rows = self.asserted.query(_PREP["graph_router"], initBindings={"req": self.U(step["believed"])})
        cands = {}
        for r in rows:
            a = str(r.a)
            if a in excluded or a not in self.actors:
                continue
            cands[a] = min(cands.get(a, 9), int(r.hop))
        self.timer.add("sparql", time.perf_counter() - t0)
        if cands:
            return sorted(cands, key=lambda a: (cands[a], -self.tfidf.cos(step["text"], self.actors[a]["description"]), a))[0]
        best = sorted(((self.tfidf.cos(step["text"], a["description"]), a["uri"]) for a in self.actors.values()
                       if a["uri"] not in excluded), reverse=True)
        return best[0][1] if best else None


class B4Tressoir(Orchestrator):
    """Specification-driven synthesis: LLM blueprint + string-level contract checks + trace-informed revision."""
    name = "B4"
    prov_fields = ("task_uri", "step", "actor", "time", "justification", "generated")

    def assign(self, step, excluded=(), feedback=None):
        ups = self.upstream_types(step, self.inputs_for(step)) if self.artifacts else list(step["task"]["in_types"])
        a, _ = self.llm.choose(step, list(self.actors.values()), ups, info=("meta", "history"), excluded=set(excluded),
                               key="blueprint", attempt=len(excluded))
        return a

    def pre_execute(self, plan):
        from .simllm import PROMPTS
        self.meter.charge(PROMPTS["blueprint"].format(goal=self.inst["goal"]["text"], feedback=""), 80 * len(plan))
        for rnd in range(MAX_REPAIRS + 1):
            bad = []
            prev_out = {}
            for s in plan:
                if "agent" not in s or s.get("_bad"):
                    s["agent"] = self.assign(s, excluded=s.get("_ex", set()))
                a = self.actors.get(s["agent"])
                if a is None:
                    continue
                prev_out[s["slot"]] = a["produces"]
            for s in plan:
                a = self.actors.get(s["agent"])
                s["_bad"] = False
                if a is None:
                    continue
                need = [prev_out[c] for c in s["consumes"] if c in prev_out]
                if s.get("goal_input"):
                    need.append(s["goal_input"])
                # string-level contract: exact type names, exact capability name or lexical label match
                io_ok = all(t in a["accepts"] for t in need) and a["produces"] == s["task"]["out_type"]
                lab = self.llm.label
                cap_ok = s["believed"] in a["capabilities"] or any(
                    self.tfidf.cos(lab[s["believed"]], lab.get(c, c)) > 0.5 for c in a["capabilities"])
                if not (io_ok and cap_ok):
                    s["_bad"] = True
                    s.setdefault("_ex", set()).add(s["agent"])
                    bad.append(s["slot"])
            if not bad:
                break
            self.notes["repairs"] += 1
            self.meter.charge(PROMPTS["blueprint"].format(goal=self.inst["goal"]["text"], feedback="CONTRACT VIOLATIONS: " + ",".join(bad)), 40 * len(bad))
        for s in plan:
            s.pop("_bad", None)
            s.pop("_ex", None)

    def insert_missing(self, step, queue, i):
        ok = self._llm_insert(step, queue, i)
        if ok:
            for s in queue[i:]:
                if not s.get("agent"):
                    s["agent"] = self.assign(s)
        return ok


class B5Compaas(_GraphMixin, Orchestrator):
    """SPARQL + procedural composition over asserted hierarchy; event-driven re-query; no LLM, no policy."""
    name, llm_decompose = "B5", False
    prov_fields = ("task_uri", "step", "actor", "time", "justification")

    def plan(self):
        self.build_asserted()
        return self.template_plan()

    def pre_execute(self, plan):
        self.unavailable = set()
        prod = {}
        for s in plan:
            ins = [prod[c] for c in s["consumes"] if c in prod] + ([s["goal_input"]] if s.get("goal_input") else [])
            s["_in"] = ins
            s["agent"] = self.assign(s)
            if s["agent"]:
                prod[s["slot"]] = self.actors[s["agent"]]["produces"]

    def assign(self, step, excluded=()):
        t0 = time.perf_counter()
        ins = step.get("_in") or [x["type"] for x in self.inputs_for(step).values()] or step["task"]["in_types"]
        sets = []
        for it in ins:
            rows = self.asserted.query(_PREP["compaas_candidates"], initBindings={
                "req": self.U(step["believed"]), "in": self.U(it), "reqout": self.U(step["task"]["out_type"])})
            sets.append({str(r.a): float(r.q) for r in rows})
        self.timer.add("sparql", time.perf_counter() - t0)
        cand = set(sets[0]) if sets else set()
        for s in sets[1:]:
            cand &= set(s)
        cand -= set(excluded) | self.unavailable
        if not cand:
            return None
        return max(sorted(cand), key=lambda a: sets[0][a])  # deterministic tie-break by URI

    def on_failure(self, step, actor, failure, tried):
        self.unavailable.add(actor)  # event-based: failed service removed from composition
        step["_in"] = None
        nxt = self.assign(step, excluded=set(tried))
        return ("retry", nxt) if nxt else ("abort",)


class B6OMCP(Orchestrator):
    """LLM agent + ontology-governed validation gate at every tool invocation (execution boundary)."""
    name = "B6"
    prov_fields = ("task_uri", "actor", "used", "time", "justification")

    def plan(self):
        p = super().plan()
        t0 = time.perf_counter()
        from .reasoner import classify
        res = classify(tbox(self.dom, defects=[tuple(d) for d in self.inst["modeling_defects"]]))
        self.closure = {k: set(v) for k, v in res["closure"].items()}
        self.timer.add("reasoning", time.perf_counter() - t0)
        return p

    def sub(self, c, d):
        return (self.ns + d) in self.closure.get(self.ns + c, {self.ns + c})

    def assign(self, step, excluded=(), feedback=None):
        ups = self.upstream_types(step, self.inputs_for(step))
        a, _ = self.llm.choose(step, list(self.actors.values()), ups, info=("meta",), excluded=set(excluded),
                               key="omcp", attempt=len(excluded))
        return a

    def _accepts_feedback(self):
        return True

    def gate(self, step, actor, inputs):
        t0 = time.perf_counter()
        a, t = self.actors[actor], step["task"]
        v = []
        if not any(self.sub(c, step["believed"]) for c in a["capabilities"]):
            v.append("CapabilityNotEntailed")
        if not all(any(self.sub(x["type"], acc) for acc in a["accepts"]) for x in inputs.values()):
            v.append("IOIncompatible")
        if not self.sub(a["produces"], t["out_type"]):
            v.append("IOIncompatible")
        if a["clearance"] < t["sensitivity"] or (t["pii"] and a["external"]) or a["cost"] > t["max_cost"]:
            v.append("PolicyViolation")
        if any(self.sub(step["believed"], h) for h in ("Approval",)) and a["kind"] != "HumanActor":
            v.append("ObligationViolation")
        self.timer.add("reasoning", time.perf_counter() - t0)
        return v

    def insert_missing(self, step, queue, i):
        return self._llm_insert(step, queue, i)


class B7CogniGraph(_GraphMixin, Orchestrator):
    """KG-topology activation: one LLM agent per activated capability node, message passing along
    KG edges, SHACL-style output governance on produced artifacts; no pre-commit policy checks."""
    name = "B7"
    prov_fields = ("task_uri", "step", "actor", "used", "time", "justification", "generated")

    def plan(self):
        self.build_asserted()
        p = self.template_plan()
        if p is None:
            p = self.llm.decompose(key="activation")
        for s in p:  # per-node agent spawn + inter-node message
            self.meter.charge("NODE " + s["text"] + " " * 1200, 120)
        return p

    def node_candidates(self, step):
        req = self.ns + step["believed"]
        out = []
        for a in self.actors.values():
            for c in a["capabilities"]:
                cu = self.ns + c
                if cu == req or (req in self.aclosure.get(cu, ()) and len(self.aclosure.get(cu, ())) - len(self.aclosure.get(req, ())) <= 1):
                    out.append(a)
                    break
        return out

    def assign(self, step, excluded=()):
        cands = [a for a in self.node_candidates(step) if a["uri"] not in excluded] or \
                [a for a in self.actors.values() if a["uri"] not in excluded]
        ups = self.upstream_types(step, self.inputs_for(step))
        a, _ = self.llm.choose(step, cands, ups, info=(), excluded=set(excluded), key="cogni", attempt=len(excluded))
        return a

    def post_check(self, step, artifact):
        self.meter.charge("GOVERNANCE " + " " * 800, 60)
        t0 = time.perf_counter()
        ok = self.asub(artifact["type"], step["task"]["out_type"])
        self.timer.add("shacl", time.perf_counter() - t0)
        return ok


class B8Covenant(Orchestrator):
    """NL workflow compiled to a controlled execution graph with policy/type guards and fallback edges."""
    name = "B8"
    prov_fields = ("task_uri", "step", "actor", "time", "justification")

    def assign(self, step, excluded=(), feedback=None):
        ups = self.upstream_types(step, self.inputs_for(step)) if self.artifacts else list(step["task"]["in_types"])
        a, _ = self.llm.choose(step, list(self.actors.values()), ups, info=("meta",), excluded=set(excluded),
                               key="covenant", attempt=len(excluded))
        return a

    def guard(self, step, a, need):
        t = step["task"]
        v = []
        if a["clearance"] < t["sensitivity"] or (t["pii"] and a["external"]) or a["cost"] > t["max_cost"]:
            v.append("PolicyGuard")
        if "approval" in self.llm.label.get(step["believed"], "").lower() and a["kind"] != "HumanActor":
            v.append("ObligationGuard")
        if not all(x in a["accepts"] for x in need) or a["produces"] != t["out_type"]:
            v.append("TypeGuard")
        return v

    def pre_execute(self, plan):
        from .simllm import PROMPTS
        self.meter.charge(PROMPTS["compile"].format(goal=self.inst["goal"]["text"],
                                                    policies="; ".join(self.inst["policies"]["rules"]), feedback=""),
                          70 * len(plan))
        prod = {}
        for s in plan:
            need = [prod[c] for c in s["consumes"] if c in prod] + ([s["goal_input"]] if s.get("goal_input") else [])
            ex = set()
            for rnd in range(MAX_REPAIRS + 1):
                a = self.assign(s, excluded=ex)
                if a is None or not self.guard(s, self.actors[a], need):
                    break
                ex.add(a)
                self.notes["repairs"] += 1
            s["agent"] = a
            fb = self.assign(s, excluded=ex | {a}) if a else None  # compiled fallback edge
            s["fallback"] = fb if fb and not self.guard(s, self.actors[fb], need) else None
            if a:
                prod[s["slot"]] = self.actors[a]["produces"]

    def on_failure(self, step, actor, failure, tried):
        fb = step.get("fallback")
        if fb and fb not in tried:
            return ("retry", fb)
        need = [x["type"] for x in self.inputs_for(step).values()]
        ex = set(tried)
        for _ in range(MAX_REPAIRS):  # controller only follows guard-satisfying edges
            nxt = self.assign(step, excluded=ex)
            if nxt is None:
                break
            if not self.guard(step, self.actors[nxt], need):
                return ("retry", nxt)
            ex.add(nxt)
        return ("abort",)


# =================================================================== ESAOG
class ESAOG(Orchestrator):
    """Full ESAOG: LLM proposal -> grounding -> OWL entailment (HermiT) -> A1 SPARQL eligibility
    -> SHACL workflow validation -> commit -> LangGraph/MCP execution -> PROV-O writeback ->
    provenance-informed local recomposition (A3)."""
    name = "P"
    W = dict(s=0.25, r=0.35, q=0.25, l=0.075, c=0.075, llm=0.05)

    def __init__(self, inst, seed, profile="nominal", entailment=True, shacl=True, provenance=True, writeback=True,
                 flatten=False, llm_planner=True, recompose=True, adapter="langgraph", name=None,
                 dataflow_grounding=True):
        super().__init__(inst, seed, profile)
        self.f = dict(entailment=entailment, shacl=shacl, provenance=provenance, writeback=writeback,
                      flatten=flatten, llm_planner=llm_planner, recompose=recompose, dataflow_grounding=dataflow_grounding)
        if name:
            self.name = name
        self.adapter = LangGraphAdapter() if adapter == "langgraph" else DirectAdapter()
        self.store = SemanticStore(inst, entailment=entailment, flatten=flatten, writeback=writeback, timer=self.timer,
                                   run_id=f"{inst['instance_id']}/{self.name}/{seed}")
        self.exec_of = {}

    # ---------------------------------------------------- planning (A2)
    def ground_dataflow(self, plan):
        """v1.1: dataflow edges are derived from the typed task signatures in G_K (requiresInput /
        requiresOutput): a step consumes the latest preceding planned step whose output type is
        subsumed by one of its required input types. Disagreeing LLM edges are corrected and logged."""
        if not self.f["dataflow_grounding"]:
            return plan
        for k, s in enumerate(plan):
            need = list(s["task"]["in_types"])
            cons = []
            for it in need:
                prod = [p for p in plan[:k] if self._sub(p["task"]["out_type"], it)]
                if prod:
                    cons.append(prod[-1]["slot"])
            cons = sorted(set(cons), key=[p["slot"] for p in plan].index)
            if sorted(cons) != sorted(s["consumes"]):
                self.notes["dataflow_corrections"] += 1
                s["consumes"] = cons
        return plan

    def plan(self):
        if not self.f["llm_planner"]:
            return self.template_plan()
        plan = self.llm.decompose()
        if self.inst["template_available"]:
            # A2 step 2 (grounding): proposed tasks are mapped to persistent task URIs in G_K; where G_K
            # already specifies the task's required capability, the KG -- not the LLM label -- is authoritative.
            gk = {t["slot"]: t["required_capability"] for t in self.inst["required_tasks"]}
            for s in plan:
                if s["believed"] != gk[s["slot"]]:
                    self.notes["grounding_corrections"] += 1
                    s["believed"] = gk[s["slot"]]
        return self.ground_dataflow(plan)

    def rank(self, step, cands, proposal=None):
        rel = self.store.reliability() if self.f["provenance"] else {}
        W = self.W
        def u(c):
            r = rel.get(c["actor"], 0.5) if self.f["provenance"] else 0.5
            return (W["s"] * c["fit"] + W["r"] * r + W["q"] * c["quality"] - W["l"] * min(c["latency"] / 6, 1)
                    - W["c"] * min(c["cost"] / 0.1, 1) + (W["llm"] if c["actor"] == proposal else 0.0))
        return sorted(cands, key=u, reverse=True)

    def steps_for_store(self, plan):
        return [dict(slot=s["slot"], task=s["task"], req=s["believed"], agent=s.get("agent"), consumes=s["consumes"],
                     goal_input=s.get("goal_input")) for s in plan]

    def pre_execute(self, plan):
        self.plan_ref = plan
        self.propose_and_verify(plan)

    def propose(self, s, excluded=()):
        if not self.f["llm_planner"]:
            return None
        ups = list(s["task"]["in_types"])
        a, _ = self.llm.choose(s, list(self.actors.values()), ups, info=("meta", "history"), excluded=set(excluded),
                               key="esaog", attempt=len(excluded))
        return a

    def propose_and_verify(self, plan):
        for s in plan:
            s["_proposal"] = self.propose(s)
            s["_cands"] = self.rank(s, self.store.eligible(s["believed"], s["task"]), s["_proposal"])
            s["_ci"] = 0
            s["agent"] = s["_cands"][0]["actor"] if s["_cands"] else None
        self.valid = False
        for rnd in range(12):
            self.store.write_workflow(self.steps_for_store(plan))
            viol = self.store.validate() if self.f["shacl"] else []
            no_cand = [s["slot"] for s in plan if s["agent"] is None]
            if not viol and not no_cand:
                self.valid = True
                break
            self.notes["repairs"] += 1
            missing = sorted({v["value"] for v in viol if v["kind"] == "MissingProducer" and v["value"]})
            if missing and self.f["llm_planner"] and rnd < MAX_REPAIRS:
                # structured violations returned to the planner (A2 step 5)
                slots = [t["slot"] for t in self.inst["required_tasks"]
                         if any(str(self.store.u(t["out_type"])) == m or
                                self._sub(t["out_type"], m.split("#")[-1]) for m in missing)]
                new = self.llm.repair_decomposition(plan, slots, key=f"esaog{rnd}")
                for s in new:
                    if "_cands" not in s:
                        s["_proposal"] = self.propose(s)
                        s["_cands"] = self.rank(s, self.store.eligible(s["believed"], s["task"]), s["_proposal"])
                        s["_ci"] = 0
                        s["agent"] = s["_cands"][0]["actor"] if s["_cands"] else None
                plan[:] = self.ground_dataflow(new)
                continue
            bad_steps = {v["slot"] for v in viol if v["kind"] in ("IOIncompatible", "ObligationViolation", "Cardinality")}
            # for dataflow violations the consumer may be at fault or its producer; advance the consumer first,
            # then the producer if the consumer runs out of candidates
            changed = False
            for s in plan:
                if s["slot"] in bad_steps:
                    changed |= self._advance(s)
            for v in viol:
                if v["kind"] == "IOIncompatible" and "upstream" in v["message"]:
                    s = next(x for x in plan if x["slot"] == v["slot"])
                    if s["agent"] is None:
                        for c in s["consumes"]:
                            up = next(x for x in plan if x["slot"] == c)
                            changed |= self._advance(up)
                            s["_ci"] = 0
                            s["agent"] = s["_cands"][0]["actor"] if s["_cands"] else None
            if not changed:
                break
        self.committed = self.store._wv
        self.graph_triples = sum(self.store.size().values())

    def _advance(self, s):
        s["_ci"] += 1
        if s["_ci"] < len(s["_cands"]):
            s["agent"] = s["_cands"][s["_ci"]]["actor"]
            return True
        s["agent"] = None
        return False

    def _sub(self, c, d):
        return str(self.store.u(d)) in self.store.closure.get(str(self.store.u(c)), ())

    # ---------------------------------------------------- execution + runtime closure (A3)
    def run(self):
        t0 = time.perf_counter()
        plan = self.plan()
        if not plan:
            self.abort_reason = "NoPlan"
            return self.finish(time.perf_counter() - t0)
        self.pre_execute(plan)
        plan = self.plan_ref
        if any(s["agent"] is None for s in plan):
            self.abort_reason = "NoCandidate"
            return self.finish(time.perf_counter() - t0)
        if not self.valid:
            self.abort_reason = "ValidationFailed"
            return self.finish(time.perf_counter() - t0)
        self.attempts_of = Counter()
        self.last_exec = {}
        remaining = list(plan)
        while remaining:
            failed = self.adapter.run(remaining, self.execute_once)
            if failed is None:
                break
            if not self.recover(failed, plan):
                break
            idx = next(i for i, s in enumerate(remaining) if s["slot"] == failed["slot"])
            remaining = remaining[idx:]
        self.graph_triples = sum(self.store.size().values())
        return self.finish(time.perf_counter() - t0)

    def execute_once(self, step):
        slot = step["slot"]
        if self.attempts_of[slot] >= MAX_ATTEMPTS:
            self.abort_reason = "AttemptsExhausted"
            return False
        self.attempts_of[slot] += 1
        inputs = self.inputs_for(step)
        actor = step["agent"]
        res = self.mcp.call(actor, slot, inputs)
        just = step.get("_why", "A1-ranked eligible candidate (SPARQL+OWL), SHACL-conformant workflow")
        e = self.store.record_execution(slot, step["task"]["uri"], actor, res["status"], failure=res["failure"],
                                        used=[x["uri"] for x in inputs.values()],
                                        artifact=res["artifact"]["uri"] if res["artifact"] else None,
                                        artifact_type=res["artifact"]["type"] if res["artifact"] else None,
                                        justification=just, replaces=self.last_exec.get(slot), started=self.clock)
        self.clock += 1
        self._prov_record(e, res)
        self.last_exec[slot] = e
        step["_last_failure"] = res["failure"]
        if res["status"] == "success":
            self.artifacts[slot] = res["artifact"]
            self.final_actor[slot] = actor
            return True
        return False

    def _prov_record(self, e, res):
        rec = {f: False for f in PROV_FIELDS}
        if e is not None:  # read back from G_E (the provenance is what is in the graph)
            g, P = self.store.ge, "http://www.w3.org/ns/prov#"
            rec["task_uri"] = (e, ESG.executesTask, None) in g
            rec["step"] = (e, ESG.executesStep, None) in g
            rec["actor"] = (e, URIRef(P + "wasAssociatedWith"), None) in g
            rec["used"] = (e, URIRef(P + "used"), None) in g
            rec["time"] = (e, URIRef(P + "startedAtTime"), None) in g
            rec["justification"] = (e, ESG.decisionJustification, None) in g
            rec["generated"] = res["status"] != "success" or (None, URIRef(P + "wasGeneratedBy"), e) in g
            rec["replaces"] = (e, ESG.isRecovery, None) not in g or (e, ESG.replacesExecution, None) in g
        self.prov.append(rec)

    def recover(self, step, plan):
        failure = step.get("_last_failure")
        slot = step["slot"]
        if self.attempts_of[slot] >= MAX_ATTEMPTS:
            self.abort_reason = "AttemptsExhausted"
            return False
        if failure == "MissingInput":
            self.abort_reason = "MissingInput"
            return False
        if failure == "AgentUnavailable":
            self.store.set_unavailable(step["agent"])  # runtime state update in G_C (M4 disables)
        if not self.f["recompose"]:
            step["_why"] = "retry (recomposition disabled)"
            return True  # M7: retry same assignment
        # A3: re-run A1 for the affected task, rank with provenance, re-validate with SHACL, new version
        self.notes["recompositions"] += 1
        bad = self.store.failed_for_task(step["task"]["uri"])  # non-transient failures recorded in G_E
        cands = self.rank(step, self.store.eligible(step["believed"], step["task"], exclude=bad))
        prev_v = self.store._wv
        old = step["agent"]
        chosen = None
        for c in cands[:5]:
            step["agent"] = c["actor"]
            self.store.write_workflow(self.steps_for_store(plan), revision_of=prev_v)
            viol = self.store.validate() if self.f["shacl"] else []
            if not viol:
                chosen = c["actor"]
                break
        if chosen is None:
            step["agent"] = old
            self.abort_reason = f"Unrecovered:{failure}"
            return False
        step["_why"] = f"recomposition after {failure} of {old.split('#')[-1]}; provenance-ranked"
        return True


ABLATIONS = {
    "M0": {},
    "M1": dict(entailment=False),
    "M2": dict(shacl=False),
    "M3": dict(provenance=False),
    "M4": dict(writeback=False),
    "M5": dict(flatten=True),
    "M6": dict(llm_planner=False),
    "M7": dict(recompose=False),
}

BASELINES = {"B0": B0Static, "B1": B1LLMOnly, "B2": B2Embedding, "B3": B3GraphRouter, "B4": B4Tressoir,
             "B5": B5Compaas, "B6": B6OMCP, "B7": B7CogniGraph, "B8": B8Covenant}


def make(method, inst, seed, profile="nominal", adapter="langgraph"):
    if method in BASELINES:
        return BASELINES[method](inst, seed, profile)
    if method == "P":
        return ESAOG(inst, seed, profile, adapter=adapter)
    if method in ABLATIONS:
        return ESAOG(inst, seed, profile, adapter=adapter, name=method, **ABLATIONS[method])
    if method == "P10":  # ESAOG v1.0: LLM-proposed dataflow edges used as grounded (first real-LLM run)
        return ESAOG(inst, seed, profile, adapter=adapter, name="P10", dataflow_grounding=False)
    raise KeyError(method)
