"""SOST benchmark generator.

Generates base workflow instances for three domains and, for each, the eight SOST
variants N/L/H/I/P/F/R/C. All perturbations are stored declaratively in the instance
JSON so that every method receives byte-identical inputs.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .domains import DOMAINS, HUMAN_ONLY_CAPS, cap_index
from . import oracle

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark" / "sost_benchmark.json"
VARIANTS = ["N", "L", "H", "I", "P", "F", "R", "C"]
BASE_PER_DOMAIN = 8
MASTER_SEED = 20260926
DEFECT_RATE = 0.12
MAX_COST = 0.10


def _descendants(children, c):
    out, stack = [], [(x, 1) for x in children.get(c, [])]
    while stack:
        x, d = stack.pop()
        out.append((x, d))
        stack += [(y, d + 1) for y in children.get(x, [])]
    return out


def _root(parent, c):
    while parent.get(c):
        c = parent[c]
    return c


class Gen:
    def __init__(self, domain, rng):
        self.domain, self.rng = domain, rng
        self.d = DOMAINS[domain]
        self.parent, self.children, self.label, self.alt, self.gloss = cap_index(domain)
        self.n = 0
        self.equiv = {}
        for a, b in self.d["equiv"]:
            self.equiv.setdefault(a, []).append(b)
            self.equiv.setdefault(b, []).append(a)

    def uri(self, stem):
        self.n += 1
        return f"{self.d['ns']}{stem}_{self.n:03d}"

    def agent(self, stem, caps, accepts, produces, desc, quality, cost, latency, rel, clearance=2,
              external=False, kind="AutomatedService", role="other"):
        u = self.uri(stem)
        return dict(uri=u, name=u.split("#")[1], kind=kind, capabilities=list(caps), accepts=list(accepts),
                    produces=produces, clearance=clearance, external=external, cost=round(cost, 4),
                    latency=round(latency, 2), quality=round(quality, 3), description=desc,
                    mcp_tool=f"mcp://{self.d['prefix']}/{u.split('#')[1].lower()}", _rel=rel, _role=role)

    def jitter(self, x, s=0.03):
        return max(0.01, x + self.rng.uniform(-s, s))


def slot_text(slot, topic):
    return slot["text"].format(topic=topic)


def req_text(base_text, req, domain):
    """Task description that names the required capability whenever it is more specific than a root
    capability (a workflow specification states what kind of step is required)."""
    parent, _, label, _, _ = cap_index(domain)
    if not parent.get(req):
        return base_text
    lab = label[req]
    return base_text.replace(" for ", f" using {lab} for ", 1) if " for " in base_text else f"{base_text} using {lab}"


def build_base(domain, b, rng):
    g = Gen(domain, rng)
    d = g.d
    topic = d["topics"][b % len(d["topics"])]
    slots = [copy.deepcopy(s) for s in d["slots"]]
    # optional last slot dropped in half of the bases -> workflows of 5 or 6 tasks
    optional = {"literature": "verify", "software": "deps", "enterprise": "archive"}[domain]
    if rng.random() < 0.5:
        slots = [s for s in slots if s["id"] != optional]
    tasks = []
    ids = {s["id"] for s in slots}
    for s in slots:
        req = rng.choice(s["req"])
        in_types = [s["in_type"]] + ([s["in_type2"]] if s.get("in_type2") else [])
        inputs = [i for i in s["inputs"] if i == "goal" or i in ids]
        pii = s["id"] in d["pii_slots"]
        tasks.append(dict(slot=s["id"], uri=f"{d['ns']}task_{s['id']}", required_capability=req,
                          text=req_text(slot_text(s, topic), req, domain), base_text=slot_text(s, topic), inputs=inputs, in_types=in_types, out_type=s["out"],
                          sub_out=s.get("sub_out"), bad_out=s["bad_out"], bad_in=s["bad_in"],
                          sensitivity=2 if pii else rng.choice([1, 1, 2]), pii=pii, max_cost=MAX_COST))
    last = tasks[-1]
    # goal output: the sink of the DAG
    consumed = {i for t in tasks for i in t["inputs"]}
    sinks = [t for t in tasks if t["slot"] not in consumed]
    goal = dict(uri=f"{d['ns']}goal_{domain}_{b:02d}", text=f"{d['goal_verb']} {topic}.",
                provides_input=tasks[0]["in_types"][0], requests_output=[t["out_type"] for t in sinks])
    agents = []
    canonical = {}
    for t in tasks:
        agents += slot_agents(g, t, canonical)
    # domain-wide agents
    gc = d["generic_cap"]
    agents.append(g.agent("generic", [gc], [tasks[0]["in_types"][0]], tasks[0]["bad_out"],
                          f"General purpose assistant for {g.label[gc]} and quick answers about {topic}.",
                          0.75, 0.005, 1.0, 0.97, clearance=1, external=True, role="generic"))
    prem_t = rng.choice(tasks)
    agents.append(g.agent("premium", [prem_t["required_capability"]], prem_t["in_types"], prem_t["out_type"],
                          f"Premium enterprise-grade {g.label[prem_t['required_capability']]} with guaranteed accuracy "
                          f"to {prem_t['text']}.", 0.98, 0.35, 2.0, 0.99, clearance=3, role="premium"))
    if domain == "enterprise" and any(t["slot"] == "approve" for t in tasks):
        t = next(t for t in tasks if t["slot"] == "approve")
        agents.append(g.agent("autoapprove", ["Approval"], t["in_types"], t["out_type"],
                              f"Automated approval engine that can instantly {t['text']}.", 0.9, 0.005, 0.3, 0.99,
                              clearance=3, role="autoapprove"))
    return dict(domain=domain, base_id=f"{d['prefix']}-{b:02d}", topic=topic, goal=goal, tasks=tasks,
                agents=agents, canonical=canonical, template_available=(b % 4 != 3)), g


def slot_agents(g: Gen, t, canonical):
    R = t["required_capability"]
    lab = g.label
    human = any(oracle.subsumed(g.domain, R, h) for h in HUMAN_ONLY_CAPS)
    kind = "HumanActor" if human else "AutomatedService"
    words = t["text"].split(" for ")[0].split(" on ")[0]
    out = []
    can = g.agent(f"{t['slot']}_agent", [R], t["in_types"], t["out_type"],
                  f"{lab[R].capitalize()} agent that can {words}.", g.jitter(0.90), g.jitter(0.02, 0.005),
                  g.jitter(2.0, 0.3), 0.97, clearance=max(2, t["sensitivity"]), kind=kind, role="canonical")
    canonical[t["slot"]] = can["uri"]
    out.append(can)
    desc = _descendants(g.children, R)
    alt_cap = desc[0][0] if desc else R
    out.append(g.agent(f"{t['slot']}_alt", [alt_cap], t["in_types"], t["sub_out"] or t["out_type"],
                       f"{lab[alt_cap].capitalize()} service: {g.gloss[alt_cap]}.", g.jitter(0.86), g.jitter(0.03, 0.005),
                       g.jitter(2.6, 0.3), 0.95, clearance=3, kind=kind, role="alt"))
    alt2_cap = desc[-1][0] if len(desc) > 1 else R
    out.append(g.agent(f"{t['slot']}_cloud", [alt2_cap], t["in_types"], t["out_type"],
                       f"Cloud-hosted {lab[alt2_cap]} API ({g.gloss[alt2_cap]}).", g.jitter(0.80), g.jitter(0.012, 0.003),
                       g.jitter(3.4, 0.3), 0.93, clearance=2, external=True, kind=kind, role="alt2"))
    p = g.parent.get(R)
    if p:
        sibs = [s for s in g.children.get(p, []) if s != R and not oracle.subsumed(g.domain, s, R)]
        if sibs:
            s = g.rng.choice(sibs)
            out.append(g.agent(f"{t['slot']}_sib", [s], t["in_types"], t["out_type"],
                               f"{lab[s].capitalize()} agent: {g.gloss[s]} to {words}.", g.jitter(0.88),
                               g.jitter(0.02, 0.005), g.jitter(2.2, 0.3), 0.96, kind=kind, role="sibling"))
        out.append(g.agent(f"{t['slot']}_general", [p], t["in_types"], t["out_type"],
                           f"General {lab[p]} assistant.", g.jitter(0.84), g.jitter(0.015, 0.004),
                           g.jitter(1.8, 0.3), 0.95, kind=kind, role="overgeneral"))
    return out


# ---------------------------------------------------------------- perturbations
def _task(inst, slot):
    return next(t for t in inst["tasks"] if t["slot"] == slot)


def _agents_for(inst, slot, role=None):
    return [a for a in inst["agents"] if a["_slot"] == slot and (role is None or a["_role"] == role)]


def perturb_L(inst, g, slot, changes):
    t = _task(inst, slot)
    R = t["required_capability"]
    cap = g.rng.choice([g.d["distractor_cap"], g.d["generic_cap"]])
    silent = g.rng.random() < 0.5
    a = g.agent(f"{slot}_expert", [cap], t["in_types"], t["out_type"] if silent else t["bad_out"],
                f"Expert agent to {t['text']}. Best-in-class {g.label[R]} results, fast and accurate.",
                0.95, 0.01, 1.2, 0.98, clearance=3, role="lexical_distractor")
    a["_slot"] = slot
    inst["agents"].append(a)
    changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], note="lexically attractive, capability-invalid"))
    for v in _agents_for(inst, slot):
        if v["_role"] in ("canonical", "alt", "alt2"):
            c = v["capabilities"][0]
            v["description"] = f"Service {v['name'].split('_')[-1]}: {g.gloss[c]}."
            changes.append(dict(op="rewrite_description", actor=v["uri"], note="terse jargon, low lexical overlap"))


def perturb_Hsub(inst, g, slot, changes):
    t = _task(inst, slot)
    G = _root(g.parent, t["required_capability"])
    desc = _descendants(g.children, G)
    if not desc:
        return False
    t["required_capability"] = G
    t["text"] = req_text(t["base_text"], G, inst["domain"])  # general requirement -> generic wording
    changes.append(dict(op="set_requirement", slot=slot, capability=G, note="general requirement"))
    for v in list(_agents_for(inst, slot)):
        if True:  # registry evolution: every previous actor for this task is retired
            inst["agents"].remove(v)
            changes.append(dict(op="retire_actor", actor=v["uri"]))
    deep = sorted(desc, key=lambda x: -x[1])
    picks = [deep[0][0]] + ([deep[1][0]] if len(deep) > 1 else [])
    for i, c in enumerate(picks):
        a = g.agent(f"{slot}_spec", [c], t["in_types"], t["sub_out"] or t["out_type"],
                    f"{g.label[c].capitalize()} module ({g.gloss[c]}).", 0.9 - 0.04 * i, 0.02 + 0.01 * i,
                    2.2 + 0.5 * i, 0.96, clearance=3, kind=_kind_for(g, G), role="specialised")
        a["_slot"] = slot
        inst["agents"].append(a)
        changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], capability=c, note="valid only via subsumption"))
    cap = g.d["distractor_cap"]
    a = g.agent(f"{slot}_lookalike", [cap], t["in_types"], t["out_type"],
                f"Agent to {t['text']} with {g.label[G]} expertise.", 0.93, 0.01, 1.5, 0.98, clearance=3,
                role="lexical_distractor")
    a["_slot"] = slot
    inst["agents"].append(a)
    changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], note="lexical look-alike, invalid capability"))
    return True


def _kind_for(g, R):
    return "HumanActor" if any(oracle.subsumed(g.domain, R, h) for h in HUMAN_ONLY_CAPS) else "AutomatedService"


def perturb_Heq(inst, g, changes, exclude):
    cands = []
    for t in inst["tasks"]:
        if t["slot"] in exclude:
            continue
        for opt in [o for s in g.d["slots"] if s["id"] == t["slot"] for o in s["req"]]:
            if opt in g.equiv:
                cands.append((t, opt))
    if not cands:
        return None
    t, R = g.rng.choice(cands)
    slot = t["slot"]
    E = g.equiv[R][0]
    t["required_capability"] = R
    t["text"] = req_text(t["base_text"], R, inst["domain"])
    changes.append(dict(op="set_requirement", slot=slot, capability=R, note="specific requirement"))
    for v in list(_agents_for(inst, slot)):
        inst["agents"].remove(v)
        changes.append(dict(op="retire_actor", actor=v["uri"]))
    kind = _kind_for(g, R)
    a = g.agent(f"{slot}_equiv", [E], t["in_types"], t["out_type"], f"{g.label[E].capitalize()} ({g.gloss[E]}).",
                0.88, 0.02, 2.4, 0.96, clearance=3, kind=kind, role="equivalent")
    a["_slot"] = slot
    inst["agents"].append(a)
    changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], capability=E, note="valid only via owl:equivalentClass"))
    p = g.parent.get(R)
    sibs = [s for s in g.children.get(p, []) if s != R and not oracle.subsumed(g.domain, s, R)] if p else []
    s = g.rng.choice(sibs) if sibs else g.d["distractor_cap"]
    a = g.agent(f"{slot}_sibling", [s], t["in_types"], t["out_type"],
                f"{g.label[s].capitalize()} agent to {t['text']}.", 0.93, 0.015, 1.8, 0.97, clearance=3, kind=kind,
                role="sibling_distractor")
    a["_slot"] = slot
    inst["agents"].append(a)
    changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], capability=s, note="sibling capability, invalid"))
    if p:
        a = g.agent(f"{slot}_parent", [p], t["in_types"], t["out_type"], f"General {g.label[p]} agent.", 0.86, 0.015,
                    1.9, 0.96, clearance=3, kind=kind, role="overgeneral")
        a["_slot"] = slot
        inst["agents"].append(a)
    return slot


def perturb_I(inst, g, slot, changes):
    t = _task(inst, slot)
    can = _agents_for(inst, slot, "canonical")[0]
    can["produces"] = t["bad_out"]
    can["description"] += " (v2 API)"
    changes.append(dict(op="change_output_type", actor=can["uri"], to=t["bad_out"], note="version update breaks dataflow"))
    a = g.agent(f"{slot}_fast", [t["required_capability"]], [t["bad_in"]], t["out_type"],
                f"Fast {g.label[t['required_capability']]} agent to {t['text']}.", 0.94, 0.01, 1.0, 0.98,
                clearance=3, kind=can["kind"], role="io_distractor")
    a["_slot"] = slot
    inst["agents"].append(a)
    changes.append(dict(op="add_actor", slot=slot, actor=a["uri"], note=f"input type {t['bad_in']} not produced upstream"))


def perturb_P(inst, g, slot, changes, mode=None):
    t = _task(inst, slot)
    can = _agents_for(inst, slot, "canonical")[0]
    modes = ["clearance", "prohibition", "budget"]
    if inst["domain"] == "enterprise" and any(x["slot"] == "approve" for x in inst["tasks"]):
        modes.append("obligation")
    mode = mode or g.rng.choice(modes)
    if mode == "clearance":
        t["sensitivity"] = 3
        can["clearance"] = 2
        for a in _agents_for(inst, slot):
            if a["_role"] == "alt":
                a["clearance"] = 3
        changes.append(dict(op="raise_sensitivity", slot=slot, to=3, note="canonical actor clearance 2"))
    elif mode == "prohibition":
        t["pii"] = True
        can["external"] = True
        can["description"] = can["description"].rstrip(".") + ", cloud-hosted."
        changes.append(dict(op="mark_pii", slot=slot, note="canonical actor externally hosted -> prohibited"))
    elif mode == "budget":
        can["cost"] = 0.25
        changes.append(dict(op="raise_cost", actor=can["uri"], to=0.25, note="exceeds task budget 0.10"))
    else:
        slot = "approve"
        for a in inst["agents"]:
            if a["_role"] == "autoapprove":
                a["quality"], a["description"] = 0.97, a["description"] + " Recommended default approver."
        changes.append(dict(op="promote_actor", role="autoapprove", note="automated approver violates human-approval obligation"))
    return mode, slot


def perturb_R(inst, g, slot, changes):
    can = _agents_for(inst, slot, "canonical")[0]
    can["_rel"] = 0.30
    can["quality"] = 0.93
    for a in _agents_for(inst, slot, "alt"):
        a["_rel"] = 0.96
    changes.append(dict(op="set_hidden_reliability", actor=can["uri"], to=0.30, note="history reveals unreliability"))


def build_instance(base, g, variant, rng):
    inst = copy.deepcopy(base)
    g.rng = rng
    for a in inst["agents"]:
        a.setdefault("_slot", _slot_of(inst, a))
    changes, events = [], []
    targets = [t["slot"] for t in inst["tasks"]]
    rng.shuffle(targets)
    target_slots = []
    if variant == "L":
        perturb_L(inst, g, targets[0], changes); target_slots = [targets[0]]
    elif variant == "H":
        s1 = next(s for s in targets if perturb_Hsub(inst, g, s, changes))
        s2 = perturb_Heq(inst, g, changes, exclude={s1})
        target_slots = [s1] + ([s2] if s2 else [])
    elif variant == "I":
        perturb_I(inst, g, targets[0], changes); target_slots = [targets[0]]
    elif variant == "P":
        mode, s = perturb_P(inst, g, targets[0], changes); target_slots = [s]
    elif variant == "F":
        s = targets[0]
        events.append(dict(type="fail_assigned_agent", slot=s, attempt=1, failure="AgentUnavailable", effect="remove_actor"))
        target_slots = [s]
    elif variant == "R":
        perturb_R(inst, g, targets[0], changes); target_slots = [targets[0]]
    elif variant == "C":
        s1 = next(s for s in targets if perturb_Hsub(inst, g, s, changes))
        rest = [s for s in targets if s != s1]
        mode, s2 = perturb_P(inst, g, rest[0], changes)
        s3 = next(s for s in rest if s not in (s1, s2))
        events.append(dict(type="fail_assigned_agent", slot=s3, attempt=1, failure="AgentUnavailable", effect="remove_actor"))
        target_slots = [s1, s2, s3]
    # provenance history: 10 prior executions per actor, consistent with hidden reliability
    hist = []
    for a in inst["agents"]:
        k = sum(rng.random() < a["_rel"] for _ in range(10))
        if a["_role"] == "canonical" and variant == "R":
            k = 3
        if a["_role"] == "alt" and variant == "R":
            k = 9
        hist.append(dict(actor=a["uri"], executions=10, successes=k))
    # ontology modelling defects (applied to every semantic method identically)
    defects = []
    if rng.random() < DEFECT_RATE:
        parent, _, _, _, _ = cap_index(inst["domain"])
        eq = dict(DOMAINS[inst["domain"]]["equiv"])
        cand = []
        for a in inst["agents"]:
            if a["_slot"] in target_slots or rng.random() < 0.3:
                for c in a["capabilities"]:
                    if parent.get(c):
                        cand.append([c, parent[c]])
                    elif c in eq.values():
                        cand.append([c, next(k for k, v in eq.items() if v == c)])
        if cand:
            defects.append(rng.choice(cand))
    return finalize(inst, variant, changes, events, target_slots, hist, defects)


def _slot_of(inst, a):
    for t in inst["tasks"]:
        if a["uri"].split("#")[1].startswith(t["slot"] + "_"):
            return t["slot"]
    return None


def finalize(inst, variant, changes, events, target_slots, hist, defects):
    dom = inst["domain"]
    tasks = inst["tasks"]
    valid = {}
    for t in tasks:
        valid[t["slot"]] = [a["uri"] for a in inst["agents"] if oracle.valid_assignment(dom, a, t)]
    checks = []
    for t in tasks:
        for a in inst["agents"]:
            if a["_slot"] != t["slot"]:
                continue
            if not oracle.cap_ok(dom, a, t["required_capability"]):
                checks.append(dict(task=t["slot"], actor=a["uri"], check="CapabilityNotEntailed"))
            if not oracle.input_ok(dom, a, t["in_types"]) or not oracle.output_ok(dom, a, t["out_type"]):
                checks.append(dict(task=t["slot"], actor=a["uri"], check="IOIncompatible"))
            for v in oracle.policy_violations(dom, a, t):
                checks.append(dict(task=t["slot"], actor=a["uri"], check=v))
    oracle_block = dict(true_reliability={a["uri"]: a["_rel"] for a in inst["agents"]},
                        roles={a["uri"]: a["_role"] for a in inst["agents"]},
                        slot_of={a["uri"]: a["_slot"] for a in inst["agents"]})
    actors = [{k: v for k, v in a.items() if not k.startswith("_")} for a in inst["agents"]]
    return dict(
        instance_id=f"{inst['base_id']}-{variant}", base_id=inst["base_id"], domain=dom, variant=variant,
        goal=inst["goal"], template_available=inst["template_available"],
        required_tasks=[{k: v for k, v in t.items() if k not in ("bad_out", "bad_in", "sub_out", "base_text")} for t in tasks],
        available_actors=actors,
        policies=dict(rules=["clearanceLevel(actor) >= sensitivityLevel(task)",
                             "not (handlesDataCategory(task, PersonalData) and hostedExternally(actor))",
                             "costPerCall(actor) <= budgetLimit(task)",
                             "requiredCapability(task) SubClassOf Approval -> actor a HumanActor"]),
        perturbation=dict(family=variant, target_slots=target_slots, changes=changes, runtime_events=events),
        valid_alternatives=valid, provenance_history=hist, modeling_defects=defects, expected_checks=checks,
        static_workflow=inst["canonical"], oracle=oracle_block)


def generate(path=BENCH):
    instances = []
    for di, dom in enumerate(DOMAINS):
        for b in range(BASE_PER_DOMAIN):
            rng = random.Random(MASTER_SEED + 1000 * di + b)
            base, g = build_base(dom, b, rng)
            for a in base["agents"]:
                a["_slot"] = _slot_of(base, a)
            for vi, v in enumerate(VARIANTS):
                vr = random.Random(MASTER_SEED + 1000 * di + 100 * b + vi + 7)
                g2 = copy.copy(g)
                instances.append(build_instance(base, g2, v, vr))
    for inst in instances:  # sanity: every slot has at least one valid alternative
        for s, v in inst["valid_alternatives"].items():
            assert v, (inst["instance_id"], s)
    meta = dict(name="SOST", version="1.0.0", master_seed=MASTER_SEED, variants=VARIANTS,
                domains=list(DOMAINS), base_per_domain=BASE_PER_DOMAIN, defect_rate=DEFECT_RATE,
                n_instances=len(instances))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta=meta, instances=instances), indent=1))
    return instances


def load(path=BENCH):
    return json.loads(Path(path).read_text())["instances"]


if __name__ == "__main__":
    inst = generate()
    print(len(inst), "instances")
