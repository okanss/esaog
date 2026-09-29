"""One-command experiment runner: every method x every SOST instance x every seed.

Outputs (raw, per run):  results/raw/runs.jsonl, results/raw/runs.csv
Manifest:               results/raw/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from esaog import benchmark  # noqa: E402
from esaog.methods import ABLATIONS, BASELINES, make  # noqa: E402
from esaog.ontology import ONTOLOGY_VERSION, tbox  # noqa: E402
from esaog.reasoner import classify  # noqa: E402

METHODS = list(BASELINES) + ["P"] + [m for m in ABLATIONS if m != "M0"]
SEEDS = [0, 1, 2, 3, 4]
RAW = ROOT / "results" / "raw"
_INST = None


def _init(instances):
    global _INST
    warnings.filterwarnings("ignore")
    _INST = {i["instance_id"]: i for i in instances}


def _job(args):
    method, iid, seed, profile = args
    inst = _INST[iid]
    try:
        r = make(method, inst, seed, profile=profile).run()
        r["profile"] = profile
        return r
    except Exception as e:  # never silently drop a run
        import traceback
        return dict(method=method, instance_id=iid, seed=seed, profile=profile, error=repr(e),
                    tb=traceback.format_exc())


def prewarm(instances):
    """Classify every distinct TBox once (serially) so that parallel workers only read the cache.
    Cold HermiT times are stored in the cache files and reported separately."""
    seen = set()
    for i in instances:
        d = tuple(tuple(x) for x in i["modeling_defects"])
        for flat in (False, True):
            k = (i["domain"], d, flat)
            if k not in seen:
                seen.add(k)
                classify(tbox(i["domain"], defects=list(d), flatten=flat))
    return len(seen)


def tree_hash():
    h = hashlib.sha256()
    for sub in ("src", "ontology", "shapes", "queries", "prompts", "scripts"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:16]


def versions():
    import importlib.metadata as md
    out = {}
    for pkg in ("rdflib", "pyshacl", "owlready2", "langgraph", "numpy", "scipy", "pandas", "statsmodels", "matplotlib"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "n/a"
    import subprocess
    try:
        out["java"] = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    except Exception:
        out["java"] = "n/a"
    return out


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":  # deterministic set/dict iteration in all workers
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--profile", default="nominal")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--regen", action="store_true", help="regenerate the SOST benchmark JSON")
    ap.add_argument("--chunksize", type=int, default=4, help="jobs per worker task (14 = one instance's methods)")
    ap.add_argument("--bases", default="", help="comma-separated base ids to restrict the instance set (e.g. lit-00,sw-03)")
    a = ap.parse_args()
    if a.regen or not benchmark.BENCH.exists():
        benchmark.generate()
    instances = benchmark.load()
    if a.bases:
        keep = set(a.bases.split(","))
        instances = [i for i in instances if i["base_id"] in keep]
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_tbox = prewarm(instances)
    methods, seeds = a.methods.split(","), [int(s) for s in a.seeds.split(",")]
    jobs = [(m, i["instance_id"], s, a.profile) for s in seeds for i in instances for m in methods]
    print(f"{len(jobs)} runs ({len(methods)} methods x {len(instances)} instances x {len(seeds)} seeds); "
          f"{n_tbox} distinct TBoxes classified", flush=True)
    rows, errors = [], 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(instances,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=a.chunksize)):
            if "error" in r:
                errors += 1
                print("ERROR", r["method"], r["instance_id"], r["seed"], r["error"], flush=True)
                print(r["tb"], flush=True)
            rows.append(r)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["method"], r["instance_id"], r["seed"]))
    with open(RAW / f"{a.out}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = [k for k in rows[0] if k not in ("causes", "failures", "tb")]
    with open(RAW / f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["causes", "failures"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["causes"] = json.dumps(r.get("causes", {}))
            rr["failures"] = json.dumps(r.get("failures", {}))
            w.writerow(rr)
    bmeta = json.loads(benchmark.BENCH.read_text())["meta"]
    manifest = dict(
        run_name=a.out, started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), duration_s=round(time.time() - t0, 1),
        code_tree_sha256_16=tree_hash(), ontology_version=ONTOLOGY_VERSION, benchmark=bmeta,
        benchmark_sha256_16=hashlib.sha256(benchmark.BENCH.read_bytes()).hexdigest()[:16],
        methods=methods, seeds=seeds, n_runs=len(rows), errors=errors,
        llm=dict(backend=("real model " + a.profile[4:]) if a.profile.startswith("llm:") else "SimLLM (simulated; no external model called)",
                 profile=a.profile, temperature_analogue="Gumbel noise",
                 prompts="prompts/*.txt", pricing_usd_per_token=dict(input=3e-6, output=15e-6)),
        cache_policy="HermiT TBox classification cached by TBox content hash (cold time reported separately); "
                     + ("LLM completions cached by exact prompt (identical prompts from different methods receive the "
                        "identical temperature-0 completion); " if a.profile.startswith("llm:") else "no caching of LLM calls; ")
                     + "no caching of SPARQL results, SHACL reports or tool results",
        python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
        packages=versions(), workers=a.workers)
    (RAW / f"manifest_{a.out}.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(rows)} runs, {errors} errors, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
