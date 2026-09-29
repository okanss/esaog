"""Ontology/SHACL coverage statistics and evaluator unit tests over benchmark expected_checks.

For every expected check (task, actor, violation) in every SOST instance, the actor is placed
in an otherwise valid workflow and we test whether ESAOG's gates (A1 SPARQL eligibility or SHACL
workflow shapes) flag it. Valid alternatives are tested the same way to measure false exclusions
(caused by the declared ontology modelling defects).
Output: results/raw/coverage.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from rdflib import Graph, RDF, RDFS, OWL, Namespace  # noqa: E402

from esaog import benchmark  # noqa: E402
from esaog.domains import DOMAINS, cap_index  # noqa: E402
from esaog.semantic import SemanticStore  # noqa: E402

SH = Namespace("http://www.w3.org/ns/shacl#")


def ontology_stats():
    core = Graph().parse(ROOT / "ontology" / "esaog-core.ttl")
    out = dict(core=dict(classes=len(set(core.subjects(RDF.type, OWL.Class))),
                         object_properties=len(set(core.subjects(RDF.type, OWL.ObjectProperty))),
                         datatype_properties=len(set(core.subjects(RDF.type, OWL.DatatypeProperty))),
                         triples=len(core)))
    for d in DOMAINS:
        g = Graph().parse(ROOT / "ontology" / f"esaog-{d}.ttl")
        parent, _, _, _, _ = cap_index(d)

        def depth(c):
            k = 1
            while parent.get(c):
                c, k = parent[c], k + 1
            return k
        out[d] = dict(capabilities=len(DOMAINS[d]["caps"]), data_types=len(DOMAINS[d]["types"]),
                      subclass_axioms=len(list(g.triples((None, RDFS.subClassOf, None)))),
                      equivalence_axioms=len(list(g.triples((None, OWL.equivalentClass, None)))),
                      max_capability_depth=max(depth(c) for c in parent), slots=len(DOMAINS[d]["slots"]),
                      triples=len(g))
    sh = Graph().parse(ROOT / "shapes" / "esaog-workflow-shapes.ttl")
    rs = Graph().parse(ROOT / "shapes" / "esaog-registry-shapes.ttl")
    out["shapes"] = dict(workflow_node_shapes=len(set(sh.subjects(RDF.type, SH.NodeShape))),
                         workflow_sparql_constraints=len(list(sh.triples((None, SH.sparql, None)))),
                         workflow_property_constraints=len(list(sh.triples((None, SH.property, None)))),
                         registry_node_shapes=len(set(rs.subjects(RDF.type, SH.NodeShape))),
                         registry_property_constraints=len(list(rs.triples((None, SH.property, None)))),
                         sparql_queries=len(list((ROOT / "queries").glob("*.rq"))))
    return out


def flagged(store, inst, slot, actor):
    tasks = {t["slot"]: t for t in inst["required_tasks"]}
    t = tasks[slot]
    el = {e["actor"] for e in store.eligible(t["required_capability"], t)}
    if actor not in el:
        return "A1"
    steps = []
    for s in inst["required_tasks"]:
        va = inst["valid_alternatives"][s["slot"]]
        steps.append(dict(slot=s["slot"], task=s, req=s["required_capability"],
                          agent=actor if s["slot"] == slot else va[0],
                          consumes=[i for i in s["inputs"] if i != "goal"],
                          goal_input=s["in_types"][0] if "goal" in s["inputs"] else None))
    store.write_workflow(steps)
    if any(v["slot"] == slot or (v["kind"] == "IOIncompatible" and slot in [c for c in tasks[v["slot"]]["inputs"]])
           for v in store.validate() if v["slot"] in tasks):
        return "SHACL"
    return None


def one(inst):
    store = SemanticStore(inst)
    det, fn, fp = Counter(), Counter(), Counter()
    by = Counter()
    for c in inst["expected_checks"]:
        g = flagged(store, inst, c["task"], c["actor"])
        (det if g else fn)[c["check"]] += 1
        if g:
            by[f"{c['check']}->{g}"] += 1
    valid_tested = 0
    for slot, va in inst["valid_alternatives"].items():
        for a in va:
            valid_tested += 1
            if flagged(store, inst, slot, a):
                fp["defect" if inst["modeling_defects"] else "no_defect"] += 1
    return dict(det=det, fn=fn, fp=fp, by=by, valid_tested=valid_tested, defect=bool(inst["modeling_defects"]))


def main():
    inst = benchmark.load()
    with mp.get_context("fork").Pool(7) as p:
        res = p.map(one, inst)
    det, fn, fp, by = Counter(), Counter(), Counter(), Counter()
    vt = 0
    for r in res:
        det.update(r["det"]); fn.update(r["fn"]); fp.update(r["fp"]); by.update(r["by"]); vt += r["valid_tested"]
    checks = sorted(set(det) | set(fn))
    out = dict(ontology=ontology_stats(),
               detection={k: dict(expected=det[k] + fn[k], detected=det[k], rate=det[k] / (det[k] + fn[k])) for k in checks},
               detected_by=dict(by), valid_alternatives_tested=vt, false_exclusions=dict(fp),
               instances_with_defects=sum(r["defect"] for r in res))
    (ROOT / "results" / "raw" / "coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["detection"], indent=1), out["false_exclusions"], vt)


if __name__ == "__main__":
    main()
