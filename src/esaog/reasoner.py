"""DL reasoning with HermiT (via owlready2) and a cached, materialised subsumption closure.

TBox classification is cached by TBox content hash (the ontology version is unchanged
across runs of the same instance); cold and warm timings are both recorded so that the
cost of formal reasoning is reported rather than hidden (paper Sec. "Complexity and caching").
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from rdflib import Graph, RDFS, OWL, URIRef

from .ontology import ESAOG, ROOT, graph_hash

CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_MEM: dict = {}


def _hermit_classify(g: Graph) -> tuple[dict, bool]:
    import owlready2
    from owlready2 import World, sync_reasoner_hermit, OwlReadyInconsistentOntologyError

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tbox.nt")
        g2 = Graph()
        for t in g:
            if t[1] != OWL.imports:  # merged graph already contains imported modules
                g2.add(t)
        g2.serialize(path, format="nt", encoding="utf-8")
        world = World()
        onto = world.get_ontology("file://" + path).load()
        consistent = True
        try:
            with onto:
                sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        except OwlReadyInconsistentOntologyError:
            consistent = False
        closure = {}
        for cls in world.classes():
            iri = cls.iri
            anc = set()
            for a in cls.ancestors(include_self=True):
                if isinstance(a, owlready2.ThingClass):
                    anc.add(a.iri)
            for e in cls.equivalent_to:  # equivalents and their ancestors
                if isinstance(e, owlready2.ThingClass):
                    anc |= {x.iri for x in e.ancestors(include_self=True) if isinstance(x, owlready2.ThingClass)}
            closure[iri] = sorted(anc)
        world.close()
    return closure, consistent


def classify(g: Graph) -> dict:
    """Return {'closure': {cls: [superclasses incl. self]}, 'consistent', 'cold_s', 'warm_s', 'hash'}."""
    h = graph_hash(g)
    t0 = time.perf_counter()
    if h in _MEM:
        res = dict(_MEM[h])
        res["warm_s"] = time.perf_counter() - t0
        res["cache_hit"] = True
        return res
    f = CACHE / f"hermit_{h}.json"
    with _LOCK:
        if f.exists():
            res = json.loads(f.read_text())
            res["cache_hit"] = True
        else:
            closure, consistent = _hermit_classify(g)
            res = dict(closure=closure, consistent=consistent, cold_s=time.perf_counter() - t0, hash=h)
            f.write_text(json.dumps(res))
            res["cache_hit"] = False
    _MEM[h] = res
    res = dict(res)
    res["warm_s"] = time.perf_counter() - t0
    return res


def materialise(g: Graph, closure: dict) -> Graph:
    """Entailed rdfs:subClassOf triples (the inferred graph queried by SPARQL eligibility)."""
    inf = Graph()
    for c, sups in closure.items():
        for s in sups:
            inf.add((URIRef(c), RDFS.subClassOf, URIRef(s)))
    return inf


def asserted_closure(g: Graph) -> dict:
    """Closure over *asserted* rdfs:subClassOf only (no OWL semantics): used by the
    COMPAAS-like baseline (SPARQL property paths) -- equivalences are not interpreted."""
    sup = {}
    for s, o in g.subject_objects(RDFS.subClassOf):
        sup.setdefault(str(s), set()).add(str(o))
    out = {}
    for c in set(sup) | {str(o) for o in g.objects(None, RDFS.subClassOf)}:
        seen, stack = {c}, [c]
        while stack:
            x = stack.pop()
            for p in sup.get(x, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        out[c] = sorted(seen)
    return out
