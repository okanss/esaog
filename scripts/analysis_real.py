"""Tables for the real-LLM reruns (results/raw/real_<model>_<part>.jsonl).

LLM-independent configurations (B0, B5, M6) are taken from the SimLLM run with seed 0, which is
identical by construction (they never call an LLM). Writes results/tables/t12_real_*.tex and
results/real_summary.json.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analysis import ABL_NAMES, BASE, NAMES, RUNTIME, VARS, compare, fmt, fp, holm  # noqa: E402

RAW, TAB = ROOT / "results" / "raw", ROOT / "results" / "tables"
NONLLM = ["B0", "B5", "M6"]
ORDER = BASE + ["P10", "P", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
PRETTY = {"qwen3-4b": "Qwen3-4B-Instruct (Ollama)", "llama3.2-3b": "Llama-3.2-3B-Instruct (Ollama)",
          "llama3.1-8b": "Llama-3.1-8B-Instruct (Ollama)",
          "qwen2.5-14b": "Qwen2.5-14B-Instruct (Ollama)"}


def load_real():
    frames = defaultdict(list)
    for f in sorted(glob.glob(str(RAW / "real_*.jsonl"))):
        m = re.match(r"real_(.+)_(sub|rest|all)\.jsonl", Path(f).name)
        if not m:
            continue
        d = pd.DataFrame([json.loads(l) for l in open(f)])
        v11 = Path(f).with_name(Path(f).stem + "_v11.jsonl")
        if v11.exists():  # ESAOG v1.1 (dataflow grounding) and its ablations replace the v1.0 rows
            d11 = pd.DataFrame([json.loads(l) for l in open(v11)])
            old = d[d.method == "P"].copy()
            old["method"] = "P10"
            keep_old = [] if "P10" in set(d11.method) else [old]  # v11 file may already contain v1.0 rows
            d = pd.concat([d[~d.method.isin(set(d11.method))], d11] + keep_old, ignore_index=True)
        frames[m.group(1)].append(d)
    out = {}
    sim = pd.DataFrame([json.loads(l) for l in open(RAW / "runs.jsonl")])
    sim0 = sim[sim.seed == 0]
    for model, fs in frames.items():
        d = pd.concat(fs, ignore_index=True)
        err = d[d.get("error").notna()] if "error" in d else d.iloc[0:0]
        d = d[d.get("error").isna()] if "error" in d else d
        ids = set(d.instance_id)
        nl = sim0[sim0.method.isin(NONLLM) & sim0.instance_id.isin(ids)]
        out[model] = dict(df=pd.concat([d, nl], ignore_index=True), errors=len(err), n_inst=len(ids))
    return out, sim0


def main():
    real, sim0 = load_real()
    if not real:
        raise SystemExit("no real-LLM results yet")
    summary, blocks = {}, []
    for model, r in sorted(real.items()):
        df, n = r["df"], r["n_inst"]
        ids = set(df.instance_id)
        simd = sim0[sim0.instance_id.isin(ids)]
        lines, s = [], {}
        for m in ORDER:
            d = df[df.method == m]
            if d.empty:
                continue
            sd = simd[simd.method == m]
            row = dict(VWS=d.VWS.mean(), SRR=d[d.variant.isin(RUNTIME)].VWS.mean(),
                       HR=d[d.variant.isin(list("HIPFR"))].VWS.mean(), CVR=d.CVR.mean(), SAA=d.SAA.mean(),
                       tokens=(d.tokens_in + d.tokens_out).mean(), lat_llm=d.lat_llm.mean(),
                       sim_VWS=sd.VWS.mean() if len(sd) else np.nan)
            s[m] = row
            nm = (r"\textbf{ESAOG} (v1.1)" if m == "P" else "ESAOG v1.0 (LLM dataflow edges)" if m == "P10"
                  else (ABL_NAMES[m] if m.startswith("M") else NAMES[m]))
            tag = "$^\\dagger$" if m in NONLLM else ""
            lines.append(f"{nm}{tag} & {fmt(row['VWS'])} & {fmt(row['HR'])} & {fmt(row['SRR'])} & {fmt(row['SAA'])} & "
                         f"{fmt(row['CVR'])} & {row['tokens']:.0f} & {fmt(row['sim_VWS'])}" + r"\\")
        # paired tests: ESAOG vs every baseline (VWS), Holm
        comps = [(b, compare(df, "P", b)) for b in BASE if not df[df.method == b].empty]
        adj = holm([c["p"] for _, c in comps])
        for (b, c), pa in zip(comps, adj):
            c["padj"] = pa
        best = max((b for b, _ in comps), key=lambda b: s[b]["VWS"])
        cb = dict(comps)[best]
        st = Counter()
        for x in df[df.method == "P"].llm_stats:
            st.update(x or {})
        lab_tot = sum(v for k, v in st.items() if k.startswith("label_")) or 1
        summary[model] = dict(n_instances=n, errors=r["errors"], rows=s, best_baseline=best,
                              vs_best=dict(diff=cb["diff"], lo=cb["lo"], hi=cb["hi"], padj=cb["padj"], b=cb["b"], c=cb["c"]),
                              label_mix={k: v / lab_tot for k, v in st.items() if k.startswith("label_")},
                              omitted=st.get("omitted", 0), invalid_answers=st.get("invalid_actor_answer", 0),
                              real_calls=int(df.llm_real_calls.sum()), cache_hits=int(df.llm_cache_hits.sum()))
        blocks.append((model, n, lines, best, cb))
    for model, n, lines, best, cb in blocks:
        pm = PRETTY.get(model, model)
        (TAB / f"t12_real_{model}.tex").write_text(r"""\begin{table}[H]
\centering
\caption{Real-LLM rerun with """ + pm + f" ({n} SOST instances, " + ("default sampling (the model accepts no temperature parameter), reasoning effort low" if "gpt" in model else "temperature 0") + ", one run per instance). " +
            r"""H--R: VWS on SOST-H/I/P/F/R; SRR on F/R/C. Tokens = mean LLM tokens per run (real calls plus modelled overhead). Last column: SimLLM (nominal, seed 0) VWS on the same instances. $^\dagger$No LLM involved (identical to the SimLLM run). ESAOG vs strongest baseline (""" +
            f"{best}): VWS difference {cb['diff']:+.3f} [{cb['lo']:+.3f}, {cb['hi']:+.3f}], b/c = {cb['b']}/{cb['c']}, Holm-adjusted exact McNemar $p${' ' if fp(cb['padj']).startswith('$') else ' = '}{fp(cb['padj'])}." +
            r"""}
\label{tab:real_""" + model.replace(".", "") + r"""}
\scriptsize
\begin{tabular}{lccccccc}
\toprule
Method & VWS & H--R & SRR & SAA & CVR$\downarrow$ & Tokens & SimLLM VWS\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (ROOT / "results" / "real_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for model, v in summary.items():
        print(model, v["n_instances"], "err", v["errors"], "best", v["best_baseline"], {k: round(x["VWS"], 2) for k, x in v["rows"].items()},
              "vs_best", {k: round(x, 3) if isinstance(x, float) else x for k, x in v["vs_best"].items()},
              "labels", {k: round(x, 2) for k, x in v["label_mix"].items()})


if __name__ == "__main__":
    main()


def summary_table():
    """Cross-backend summary: SimLLM (nominal, 5 seeds, all instances) vs each real model."""
    real, _ = load_real()
    sim = pd.DataFrame([json.loads(l) for l in open(RAW / "runs.jsonl")])
    models = sorted(real)
    rows = BASE + ["P10", "P", "M6"]
    order = ["gpt-6-luna", "claude-haiku-4-5", "qwen3-4b", "llama3.2-3b", "llama3.1-8b", "qwen2.5-14b"]
    models = [m for m in order if m in real] + [m for m in models if m not in order]
    def head(m):
        n = real[m]["n_inst"]
        nm = {"gpt-6-luna": "gpt-6-luna", "claude-haiku-4-5": "Haiku 4.5", "qwen3-4b": "Qwen3-4B",
              "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B", "qwen2.5-14b": "Qwen2.5-14B"}.get(m, m)
        return nm + (f"$^*$" if n < 192 else "")
    hdr = " & ".join([r"\multicolumn{3}{c}{SimLLM (nominal)}"] + [r"\multicolumn{3}{c}{" + head(m) + "}" for m in models])
    sub = " & ".join(["VWS & H--R & SRR"] * (1 + len(models)))
    cm = "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(1 + len(models)))
    lines = []

    def cells(d, m):
        x = d[d.method == m]
        if x.empty:
            return ["--"] * 3
        return [fmt(x.VWS.mean()), fmt(x[x.variant.isin(list("HIPFR"))].VWS.mean()), fmt(x[x.variant.isin(RUNTIME)].VWS.mean())]
    for m in rows:
        nm = (r"\textbf{ESAOG} v1.1" if m == "P" else "ESAOG v1.0" if m == "P10" else
              (ABL_NAMES[m] if m.startswith("M") else NAMES[m]))
        c = cells(sim, m) + sum((cells(real[k]["df"], m) for k in models), [])
        lines.append(nm + " & " + " & ".join(c) + r"\\")
    (TAB / "t13_real_summary.tex").write_text(r"""\begin{table}[H]
\centering
\caption{Simulated versus real LLM proposers (Valid Workflow Success). SimLLM: 192 instances $\times$ 5 seeds; real models: 192 instances, one run per instance. H--R: SOST-H/I/P/F/R; SRR: F/R/C. The SimLLM ESAOG v1.0 cell is omitted because v1.0 and v1.1 coincide under SimLLM, which proposes correct dataflow edges (archived v1.0 run on SOST v1.0.1: 0.96; Table~\ref{tab:benchver}). B0, B5 and M6 do not call an LLM.}
\label{tab:realsummary}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "ccc" * (1 + len(models)) + r"""}
\toprule
& """ + hdr + r"""\\
""" + cm + r"""
Method & """ + sub + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


if __name__ == "__main__":
    summary_table()


def benchmark_version_table():
    """VWS on SOST v1.0.0 vs v1.0.1 with the same ESAOG v1.0 code (archived runs; no new experiments)."""
    def L(f):
        return pd.DataFrame([json.loads(l) for l in open(RAW / f)])
    sa, sb = L("superseded_v1/runs.jsonl"), L("superseded_v1_0sim/runs.jsonl")
    la, lb = L("superseded_v1/real_gpt-6-luna_all.jsonl"), L("real_gpt-6-luna_all.jsonl")
    rows = [("P", r"ESAOG (v1.0)"), ("B6", NAMES["B6"]), ("B1", NAMES["B1"]), ("M1", ABL_NAMES["M1"]), ("M2", ABL_NAMES["M2"])]
    lines = []
    for m, nm in rows:
        v = [d[d.method == m].VWS.mean() for d in (sa, sb, la, lb)]
        d1, d2 = round(v[1]-v[0], 2) + 0.0, round(v[3]-v[2], 2) + 0.0
        lines.append(f"{nm} & {v[0]:.2f} & {v[1]:.2f} & {d1:+.2f} & {v[2]:.2f} & {v[3]:.2f} & {d2:+.2f}" + r"\\")
    (TAB / "t14_benchver.tex").write_text(r"""\begin{table}[H]
\centering
\caption{Benchmark-version sensitivity (post-hoc correction). VWS on SOST v1.0.0 and the corrected v1.0.1 with the same ESAOG v1.0 code, taken from the archived runs made before the correction (SimLLM: 192 instances $\times$ 5 seeds; \texttt{gpt-6-luna}: 192 instances, one run each). No run was repeated for this table.}
\label{tab:benchver}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{SimLLM (nominal)} & \multicolumn{3}{c}{\texttt{gpt-6-luna}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & v1.0.0 & v1.0.1 & $\Delta$VWS & v1.0.0 & v1.0.1 & $\Delta$VWS\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    print("\n".join(lines))


if __name__ == "__main__":
    benchmark_version_table()
