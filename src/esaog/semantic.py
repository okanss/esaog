"""ESAOG semantic core: four named graphs (G_K, G_C, G_W, G_E) with shared URIs,
HermiT-backed entailment, SPARQL eligibility (A1), SHACL workflow validation, PROV-O
execution records and provenance-derived reliability.

The core is runtime-neutral: it never calls a tool itself; adapters do.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from pyshacl import validate as shacl_validate
from rdflib import BNode, Dataset, Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.plugins.sparql import prepareQuery

from .domains import DOMAINS
from .ontology import ESAOG, PROV, ROOT, tbox
from .reasoner import classify, materialise

Q = {p.stem: (ROOT / "queries" / p.name).read_text() for p in (ROOT / "queries").glob("*.rq")}
_PREP = {k: prepareQuery(v) for k, v in Q.items()}
SHAPES = Graph().parse(ROOT / "shapes" / "esaog-workflow-shapes.ttl", format="turtle")
REG_SHAPES = Graph().parse(ROOT / "shapes" / "esaog-registry-shapes.ttl", format="turtle")
RUN = Namespace("https://w3id.org/esaog/run/")
GK, GC, GW, GE = (URIRef(f"https://w3id.org/esaog/graph/{x}") for x in ("knowledge", "capability", "workflow", "execution"))


class Timer:
    def __init__(self):
        self.t = {"reasoning": 0.0, "sparql": 0.0, "shacl": 0.0, "graph_write": 0.0, "reasoning_cold": 0.0}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt


class SemanticStore:
    """Per-run orchestration state G_ESAOG = G_K u G_C u G_W u G_E."""

    def __init__(self, inst: dict, entailment=True, flatten=False, writeback=True, timer: Timer | None = None,
                 run_id="run"):
        self.inst, self.domain = inst, inst["domain"]
        self.ns = Namespace(DOMAINS[self.domain]["ns"])
        self.timer = timer or Timer()
        self.entailment, self.writeback = entailment, writeback
        self.run = Namespace(f"https://w3id.org/esaog/run/{run_id}/")
        self.ds = Dataset()
        self.gk, self.gc, self.gw, self.ge = (self.ds.graph(x) for x in (GK, GC, GW, GE))
        t0 = time.perf_counter()
        tb = tbox(self.domain, defects=[tuple(d) for d in inst["modeling_defects"]], flatten=flatten)
        if entailment:
            res = classify(tb)
            self.timer.add("reasoning_cold", res.get("cold_s", 0.0) if not res.get("cache_hit") else 0.0)
            self.closure = {k: set(v) for k, v in res["closure"].items()}
            self.consistent = res["consistent"]
        else:  # M1: no entailment -- only the reflexive identity is available
            self.closure = {str(c): {str(c)} for c in tb.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))}
            self.consistent = True
        self.inferred = Graph()
        for c, sups in self.closure.items():
            for s in sups:
                self.inferred.add((URIRef(c), RDFS.subClassOf, URIRef(s)))
        for c in tb.subjects(ESAOG.mustBeRealizedBy, None):
            for o in tb.objects(c, ESAOG.mustBeRealizedBy):
                self.inferred.add((c, ESAOG.mustBeRealizedBy, o))
        self.timer.add("reasoning", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self._load_knowledge()
        self._load_capabilities()
        self._load_history()
        self.timer.add("graph_write", time.perf_counter() - t0)
        self._ctr = itertools.count(1)
        self.version = 0
        self._wv = None

    # ------------------------------------------------------------------ loading
    def u(self, local):
        return self.ns[local] if not str(local).startswith("http") else URIRef(local)

    def _load_knowledge(self):
        g, goal = self.gk, self.inst["goal"]
        gu = URIRef(goal["uri"])
        g.add((gu, RDF.type, ESAOG.Goal))
        g.add((gu, RDFS.comment, Literal(goal["text"])))
        g.add((gu, ESAOG.providesInput, self.u(goal["provides_input"])))
        for o in goal["requests_output"]:
            g.add((gu, ESAOG.requestsOutput, self.u(o)))

    def _load_capabilities(self):
        g = self.gc
        for a in self.inst["available_actors"]:
            au = URIRef(a["uri"])
            g.add((au, RDF.type, ESAOG.Actor))
            g.add((au, RDF.type, ESAOG.Agent))
            g.add((au, RDF.type, ESAOG[a["kind"]]))
            for c in a["capabilities"]:
                g.add((au, ESAOG.hasCapability, self.u(c)))
            for t in a["accepts"]:
                g.add((au, ESAOG.acceptsInput, self.u(t)))
            g.add((au, ESAOG.producesOutput, self.u(a["produces"])))
            g.add((au, ESAOG.clearanceLevel, Literal(a["clearance"], datatype=XSD.integer)))
            g.add((au, ESAOG.hostedExternally, Literal(a["external"])))
            g.add((au, ESAOG.costPerCall, Literal(a["cost"], datatype=XSD.decimal)))
            g.add((au, ESAOG.expectedLatency, Literal(a["latency"], datatype=XSD.decimal)))
            g.add((au, ESAOG.declaredQuality, Literal(a["quality"], datatype=XSD.decimal)))
            g.add((au, ESAOG.available, Literal(True)))
            g.add((au, ESAOG.availableAs, Literal(a["mcp_tool"])))
            g.add((au, ESAOG.description, Literal(a["description"])))

    def _load_history(self):
        """Import prior executions (provenance_history) into G_E as PROV activities."""
        g = self.ge
        for h in self.inst["provenance_history"]:
            au = URIRef(h["actor"])
            for i in range(h["executions"]):
                e = self.run[f"hist/{au.split('#')[1]}/{i}"]
                g.add((e, RDF.type, ESAOG.Execution))
                g.add((e, PROV.wasAssociatedWith, au))
                g.add((e, ESAOG.hasState, ESAOG.Succeeded if i < h["successes"] else ESAOG.Failed))

    # ------------------------------------------------------------------ A4 registration
    def register_actor(self, manifest: dict) -> tuple[bool, list]:
        """Controlled semantic registration (A4): SHACL registry shapes + consistency."""
        g = Graph()
        au = URIRef(manifest["uri"])
        g.add((au, RDF.type, ESAOG.Actor))
        for c in manifest.get("capabilities", []):
            g.add((au, ESAOG.hasCapability, self.u(c)))
        for t in manifest.get("accepts", []):
            g.add((au, ESAOG.acceptsInput, self.u(t)))
        if manifest.get("produces"):
            g.add((au, ESAOG.producesOutput, self.u(manifest["produces"])))
        for k, p, dt in (("clearance", ESAOG.clearanceLevel, XSD.integer), ("cost", ESAOG.costPerCall, XSD.decimal),
                         ("latency", ESAOG.expectedLatency, XSD.decimal)):
            if k in manifest:
                g.add((au, p, Literal(str(manifest[k]), datatype=dt)))
        if "external" in manifest:
            g.add((au, ESAOG.hostedExternally, Literal(manifest["external"])))
        if "mcp_tool" in manifest:
            g.add((au, ESAOG.availableAs, Literal(manifest["mcp_tool"])))
        unknown = [c for c in manifest.get("capabilities", []) if str(self.u(c)) not in self.closure]
        conforms, rg, _ = shacl_validate(g, shacl_graph=REG_SHAPES)
        errs = [str(m) for m in rg.objects(None, URIRef("http://www.w3.org/ns/shacl#resultMessage"))]
        errs += [f"UnknownCapability {c}" for c in unknown]
        if conforms and not unknown:
            for t in g:
                self.gc.add(t)
            self.gc.add((au, ESAOG.available, Literal(True)))
            return True, []
        return False, errs

    # ------------------------------------------------------------------ A1 eligibility
    def eligible(self, req: str, task: dict, exclude=()):
        t0 = time.perf_counter()
        union = self.gc + self.inferred
        rows = union.query(_PREP["eligibility"], initBindings={
            "req": self.u(req), "sens": Literal(task["sensitivity"], datatype=XSD.integer),
            "budget": Literal(task["max_cost"], datatype=XSD.decimal), "pii": Literal(bool(task["pii"]))})
        out = {}
        for r in rows:
            a = str(r.a)
            if a in exclude:
                continue
            fit = self.semantic_fit(str(r.cap), str(self.u(req)))
            if a not in out or fit > out[a]["fit"]:
                out[a] = dict(actor=a, cap=str(r.cap), fit=fit, quality=float(r.q), cost=float(r.cost), latency=float(r.lat))
        self.timer.add("sparql", time.perf_counter() - t0)
        return list(out.values())

    def semantic_fit(self, cap, req):
        if cap == req or (req in self.closure.get(cap, ()) and cap in self.closure.get(req, ())):
            return 1.0
        depth = len(self.closure.get(cap, ())) - len(self.closure.get(req, ()))
        return max(0.4, 1.0 - 0.2 * max(depth, 1))

    def reliability(self):
        t0 = time.perf_counter()
        rel = {}
        for r in self.ge.query(_PREP["reliability"]):
            s, n = int(r.succ), int(r.n)
            rel[str(r.a)] = (1 + s) / (2 + n)  # Beta(1,1) posterior mean
        self.timer.add("sparql", time.perf_counter() - t0)
        return rel

    def failed_for_task(self, task_uri):
        """Actors whose execution of this task failed non-transiently (type mismatch, unavailable)."""
        t0 = time.perf_counter()
        q = """PREFIX esaog: <https://w3id.org/esaog/core#> PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT DISTINCT ?a WHERE { ?e esaog:executesTask ?t ; prov:wasAssociatedWith ?a ; esaog:failedWith ?f .
        ?f a ?k . FILTER(?k IN (esaog:TypeMismatch, esaog:AgentUnavailable, esaog:MissingInput)) }"""
        out = {str(r.a) for r in self.ge.query(q, initBindings={"t": URIRef(task_uri)})}
        self.timer.add("sparql", time.perf_counter() - t0)
        return out

    def set_unavailable(self, actor):
        if not self.writeback:
            return
        t0 = time.perf_counter()
        au = URIRef(actor)
        self.gc.remove((au, ESAOG.available, None))
        self.gc.add((au, ESAOG.available, Literal(False)))
        self.timer.add("graph_write", time.perf_counter() - t0)

    # ------------------------------------------------------------------ G_W
    def write_workflow(self, steps: list, revision_of=None):
        """steps: [{slot, task (dict with semantics), req, agent, consumes:[slot], goal_input:type|None}]"""
        t0 = time.perf_counter()
        self.version += 1
        wv = self.run[f"workflow/v{self.version}"]
        g = Graph()
        g.add((wv, RDF.type, ESAOG.WorkflowVersion))
        g.add((wv, ESAOG.forGoal, URIRef(self.inst["goal"]["uri"])))
        if revision_of is not None:
            g.add((wv, PROV.wasRevisionOf, revision_of))
        for s in steps:
            su = self.step_uri(s["slot"])
            tu = URIRef(s["task"]["uri"])
            g.add((wv, ESAOG.hasStep, su))
            g.add((su, RDF.type, ESAOG.ExecutableStep))
            g.add((su, ESAOG.realizesTask, tu))
            if s.get("agent"):
                g.add((su, ESAOG.assignedAgent, URIRef(s["agent"])))
            g.add((tu, RDF.type, ESAOG.Task))
            g.add((tu, ESAOG.requiresCapability, self.u(s["req"])))
            for it in s["task"]["in_types"]:
                g.add((tu, ESAOG.requiresInput, self.u(it)))
            g.add((tu, ESAOG.requiresOutput, self.u(s["task"]["out_type"])))
            g.add((tu, ESAOG.sensitivityLevel, Literal(s["task"]["sensitivity"], datatype=XSD.integer)))
            g.add((tu, ESAOG.budgetLimit, Literal(s["task"]["max_cost"], datatype=XSD.decimal)))
            if s["task"]["pii"]:
                g.add((tu, ESAOG.handlesDataCategory, ESAOG.PersonalData))
            for c in s["consumes"]:
                g.add((su, ESAOG.consumesFrom, self.step_uri(c)))
                g.add((tu, ESAOG.dependsOn, URIRef(f"{self.ns}task_{c}")))
            if s.get("goal_input"):
                g.add((su, ESAOG.consumesGoalInput, self.u(s["goal_input"])))
        # replace current workflow view; older versions stay in G_E lineage
        self.gw.remove((None, None, None))
        for t in g:
            self.gw.add(t)
        self._wv = wv
        self.timer.add("graph_write", time.perf_counter() - t0)
        return wv

    def step_uri(self, slot):
        return self.run[f"step/{slot}"]

    def validate(self):
        """SHACL workflow validation -> structured violations [{shape, focus_slot, value, message}]."""
        t0 = time.perf_counter()
        data = self.gw + self.gc + self.gk + self.inferred
        conforms, rg, _ = shacl_validate(data, shacl_graph=SHAPES, advanced=True, inference="none")
        viol = []
        SH = Namespace("http://www.w3.org/ns/shacl#")
        for r in rg.subjects(RDF.type, SH.ValidationResult):
            focus = str(rg.value(r, SH.focusNode))
            msg = str(rg.value(r, SH.resultMessage))
            val = rg.value(r, SH.value)
            viol.append(dict(kind=msg.split(":")[0], focus=focus, slot=focus.rsplit("/", 1)[-1],
                             value=str(val) if val is not None else None, message=msg))
        self.timer.add("shacl", time.perf_counter() - t0)
        return viol

    # ------------------------------------------------------------------ G_E (PROV-O)
    def record_execution(self, slot, task_uri, actor, state, failure=None, used=(), artifact=None, artifact_type=None,
                         justification="", replaces=None, started=0.0):
        if not self.writeback:
            return None
        t0 = time.perf_counter()
        e = self.run[f"exec/{next(self._ctr)}"]
        g = self.ge
        g.add((e, RDF.type, ESAOG.Execution))
        g.add((e, RDF.type, ESAOG.Invocation))
        g.add((e, ESAOG.runLocal, Literal(True)))
        g.add((e, ESAOG.executesTask, URIRef(task_uri)))
        g.add((e, ESAOG.executesStep, self.step_uri(slot)))
        g.add((e, ESAOG.inWorkflowVersion, self._wv))
        g.add((e, PROV.wasAssociatedWith, URIRef(actor)))
        g.add((e, PROV.startedAtTime, Literal(started, datatype=XSD.decimal)))
        g.add((e, ESAOG.decisionJustification, Literal(justification)))
        g.add((e, ESAOG.hasState, ESAOG.Succeeded if state == "success" else ESAOG.Failed))
        for u in used:
            g.add((e, PROV.used, URIRef(u)))
        if artifact:
            au = URIRef(artifact)
            g.add((au, RDF.type, ESAOG.Artifact))
            g.add((au, PROV.wasGeneratedBy, e))
            g.add((au, ESAOG.artifactType, self.u(artifact_type)))
            for u in used:
                g.add((au, PROV.wasDerivedFrom, URIRef(u)))
        if failure:
            f = self.run[f"failure/{e.split('/')[-1]}"]
            g.add((f, RDF.type, ESAOG[failure]))
            g.add((e, ESAOG.failedWith, f))
        if replaces is not None:
            g.add((e, ESAOG.isRecovery, Literal(True)))
            g.add((e, ESAOG.replacesExecution, replaces))
            g.add((self.run[f"recovery/{e.split('/')[-1]}"], RDF.type, ESAOG.RecoveryStrategy))
            g.add((e, ESAOG.recoveredBy, self.run[f"recovery/{e.split('/')[-1]}"]))
        self.timer.add("graph_write", time.perf_counter() - t0)
        return e

    def provenance_completeness(self):
        t0 = time.perf_counter()
        rows = list(self.ge.query(_PREP["provenance_completeness"]))
        self.timer.add("sparql", time.perf_counter() - t0)
        if not rows:
            return None
        return sum(1 for r in rows if bool(r.complete.toPython())) / len(rows)

    def size(self):
        return dict(G_K=len(self.gk), G_C=len(self.gc), G_W=len(self.gw), G_E=len(self.ge), inferred=len(self.inferred))
