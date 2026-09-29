"""Failure-to-recomposition trace example: runs full ESAOG on a compound (SOST-C) instance,
exports the four named graphs as TriG, and renders the G_E trace (queried with failure_trace.rq)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from esaog import benchmark  # noqa: E402
from esaog.methods import make  # noqa: E402
from esaog.semantic import _PREP  # noqa: E402


def main(iid=None):
    inst_all = benchmark.load()
    cands = [i for i in inst_all if i["variant"] == "C"] if iid is None else [i for i in inst_all if i["instance_id"] == iid]
    for inst in cands:
        m = make("P", inst, 0)
        r = m.run()
        if r["VWS"] and r["recompositions"] >= 1:
            break
    out = ROOT / "results" / "traces"
    out.mkdir(parents=True, exist_ok=True)
    m.store.ds.serialize(out / f"{inst['instance_id']}_P_seed0.trig", format="trig")
    rows = []
    for t in inst["required_tasks"]:
        for q in m.store.ge.query(_PREP["failure_trace"], initBindings={"task": __import__("rdflib").URIRef(t["uri"])}):
            rows.append(dict(slot=t["slot"], e=str(q.e).split("/")[-1], actor=str(q.a).split("#")[-1],
                             state=str(q.st).split("#")[-1], failure=str(q.f).split("#")[-1] if q.f else "",
                             old=str(q.old).split("/")[-1] if q.old else "", wv=str(q.wv).split("/")[-1],
                             why=str(q.j)))
    rows.sort(key=lambda x: int(x["e"]))
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(rows) + 1.0))
    for k, x in enumerate(rows):
        y = len(rows) - k
        c = "#27ae60" if x["state"] == "Succeeded" else "#c0392b"
        ax.text(0.0, y, f"exec/{x['e']}", fontsize=8, family="monospace")
        ax.text(0.11, y, f"task_{x['slot']}", fontsize=8, family="monospace")
        ax.text(0.25, y, x["actor"], fontsize=8, family="monospace")
        ax.text(0.46, y, x["state"] + (f" ({x['failure']})" if x["failure"] else ""), fontsize=8, color=c)
        ax.text(0.66, y, x["wv"], fontsize=8, family="monospace")
        ax.text(0.73, y, (f"replaces exec/{x['old']}" if x["old"] else ""), fontsize=8, color="#2c3e50")
    for j, h in enumerate(["Execution", "Task URI", "Actor", "State", "Workflow", "Provenance link"]):
        ax.text([0.0, 0.11, 0.25, 0.46, 0.66, 0.73][j], len(rows) + 1, h, fontsize=8, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1.6)
    ax.axis("off")
    ts = inst["perturbation"]["target_slots"]
    ax.set_title(f"{inst['instance_id']} (SOST-C): hierarchical requirement on task_{ts[0]}, policy conflict on task_{ts[1]}, "
                 f"injected unavailability on task_{ts[2]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "failure_trace.pdf")
    (out / "trace_example.txt").write_text("\n".join(f"{x}" for x in rows))
    print(inst["instance_id"], r["VWS"], r["recompositions"])
    for x in rows:
        print(x)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
