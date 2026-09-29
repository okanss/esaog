"""Unit tests for the semantic core (run: python -m pytest tests)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from esaog import benchmark, oracle  # noqa: E402
from esaog.methods import make  # noqa: E402
from esaog.semantic import SemanticStore  # noqa: E402

I = {i["instance_id"]: i for i in benchmark.load()}


def test_equivalence_and_subsumption_entailed():
    assert oracle.subsumed("literature", "PubMedSearch", "LiteratureSearch")
    assert oracle.subsumed("literature", "CochraneRiskOfBias2", "RiskOfBiasAssessment")  # via owl:equivalentClass
    assert not oracle.subsumed("literature", "ROBINSIAssessment", "RoB2Assessment")
    assert oracle.subsumed("enterprise", "BillDataCapture", "DocumentExtraction")


def test_a1_matches_oracle_without_defects():
    for inst in I.values():
        if inst["modeling_defects"]:
            continue
        st = SemanticStore(inst)
        for t in inst["required_tasks"]:
            el = {e["actor"] for e in st.eligible(t["required_capability"], t)}
            for a in inst["available_actors"]:
                cap = oracle.cap_ok(inst["domain"], a, t["required_capability"])
                pol = [v for v in oracle.policy_violations(inst["domain"], a, t) if v != "ObligationViolation"]
                assert (a["uri"] in el) == (cap and not pol), (inst["instance_id"], t["slot"], a["name"])
        break


def test_m1_loses_entailment():
    inst = I["lit-00-H"]
    t = next(t for t in inst["required_tasks"] if t["slot"] == "synth")
    assert SemanticStore(inst).eligible(t["required_capability"], t)
    assert not SemanticStore(inst, entailment=False).eligible(t["required_capability"], t)


def test_shacl_detects_io_incompatibility():
    inst = I["lit-00-I"]
    st = SemanticStore(inst)
    slot = inst["perturbation"]["target_slots"][0]
    bad = next(c["actor"] for c in inst["expected_checks"] if c["task"] == slot and c["check"] == "IOIncompatible")
    steps = []
    for s in inst["required_tasks"]:
        steps.append(dict(slot=s["slot"], task=s, req=s["required_capability"],
                          agent=bad if s["slot"] == slot else inst["valid_alternatives"][s["slot"]][0],
                          consumes=[i for i in s["inputs"] if i != "goal"],
                          goal_input=s["in_types"][0] if "goal" in s["inputs"] else None))
    st.write_workflow(steps)
    assert any(v["kind"] == "IOIncompatible" for v in st.validate())


def test_registration_rejects_incomplete_manifest():
    st = SemanticStore(I["sw-00-N"])
    ok, errs = st.register_actor(dict(uri="https://w3id.org/esaog/sw#new_1", capabilities=["UnitTesting"]))
    assert not ok and errs
    ok, _ = st.register_actor(dict(uri="https://w3id.org/esaog/sw#new_2", capabilities=["UnitTesting"],
                                   accepts=["SourcePatch"], produces="TestReport", clearance=2, cost=0.01,
                                   latency=1.0, external=False, mcp_tool="mcp://sw/new_2"))
    assert ok


def test_adapters_equivalent():
    for iid in ("ent-01-C", "lit-02-F", "sw-03-R"):
        a = make("P", I[iid], 0, adapter="langgraph").run()
        b = make("P", I[iid], 0, adapter="direct").run()
        for k in ("TSR", "VWS", "attempts", "SAA"):
            assert a[k] == b[k], (iid, k)


def test_uri_continuity_task_survives_failure():
    inst = I["lit-01-F"]
    m = make("P", inst, 0)
    m.run()
    slot = inst["perturbation"]["target_slots"][0]
    task_uri = next(t["uri"] for t in inst["required_tasks"] if t["slot"] == slot)
    q = """PREFIX esaog: <https://w3id.org/esaog/core#>
    SELECT ?e WHERE { ?e esaog:executesTask ?t ; esaog:runLocal true }"""
    from rdflib import URIRef
    rows = list(m.store.ge.query(q, initBindings={"t": URIRef(task_uri)}))
    assert len(rows) >= 2  # failed attempt and its replacement share the task URI
