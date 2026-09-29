"""Ground-truth semantics used ONLY by the evaluator and the simulated runtime.

The oracle uses the complete (defect-free) domain ontology; methods never see it
directly -- they see the (possibly defective) ontology shipped with each instance.
"""
from __future__ import annotations

from functools import lru_cache

from .domains import DOMAINS, HUMAN_ONLY_CAPS
from .ontology import tbox
from .reasoner import classify


@lru_cache(maxsize=None)
def true_closure(domain: str) -> dict:
    c = classify(tbox(domain))["closure"]
    ns = DOMAINS[domain]["ns"]
    out = {}
    for k, v in c.items():
        if k.startswith(ns):
            out[k[len(ns):]] = frozenset(x[len(ns):] for x in v if x.startswith(ns))
    return out


def subsumed(domain, sub, sup) -> bool:
    """True iff O |= sub SubClassOf sup (names are local names in the domain namespace)."""
    return sup in true_closure(domain).get(sub, frozenset({sub}))


def cap_ok(domain, agent, req) -> bool:
    return any(subsumed(domain, c, req) for c in agent["capabilities"])


def input_ok(domain, agent, in_types) -> bool:
    return all(any(subsumed(domain, t, a) for a in agent["accepts"]) for t in in_types)


def output_ok(domain, agent, out_type) -> bool:
    return subsumed(domain, agent["produces"], out_type)


def human_required(domain, req) -> bool:
    return any(subsumed(domain, req, h) for h in HUMAN_ONLY_CAPS)


def policy_violations(domain, agent, task) -> list[str]:
    v = []
    if agent["clearance"] < task["sensitivity"]:
        v.append("ClearanceViolation")
    if task["pii"] and agent["external"]:
        v.append("ProhibitionViolation")
    if agent["cost"] > task["max_cost"]:
        v.append("BudgetViolation")
    if human_required(domain, task["required_capability"]) and agent["kind"] != "HumanActor":
        v.append("ObligationViolation")
    return v


def valid_assignment(domain, agent, task, in_types=None) -> bool:
    in_types = in_types if in_types is not None else task["in_types"]
    return (cap_ok(domain, agent, task["required_capability"]) and input_ok(domain, agent, in_types)
            and output_ok(domain, agent, task["out_type"]) and not policy_violations(domain, agent, task))


def utility(agent, rel) -> float:
    """Oracle utility used for Recovery Regret: expected quality minus normalised cost/latency."""
    return rel * agent["quality"] - 0.5 * min(agent["cost"] / 0.1, 1.0) * 0.2 - 0.2 * min(agent["latency"] / 6.0, 1.0)
