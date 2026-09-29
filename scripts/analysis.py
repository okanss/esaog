"""Generate all tables (LaTeX) and figures from raw per-run results.

  python scripts/analysis.py            -> results/tables/*.tex, results/figures/*.pdf, results/summary.json
Statistics: cluster (instance) bootstrap 95% CIs; exact McNemar tests for paired binary outcomes;
Wilcoxon signed-rank for paired continuous outcomes; Holm correction within each table; effect
sizes (risk difference, Cohen's h, discordant-pair odds ratio, rank-biserial r).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, TAB, FIG = ROOT / "results" / "raw", ROOT / "results" / "tables", ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260926)
NBOOT = 2000
BASE = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
NAMES = {"B0": "B0 Static", "B1": "B1 LLM-only", "B2": "B2 TF-IDF lexical", "B3": "B3 Graph router",
         "B4": "B4 Tressoir-like", "B5": "B5 COMPAAS-like", "B6": "B6 O-MCP-like", "B7": "B7 CogniGraph-like",
         "B8": "B8 COVENANT-like", "P": "ESAOG"}
ABL = ["P", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
ABL_NAMES = {"P": "M0 Full", "M1": "M1 $-$OWL", "M2": "M2 $-$SHACL", "M3": "M3 $-$Provenance ranking",
             "M4": "M4 $-$Runtime KG", "M5": "M5 Flat capabilities", "M6": "M6 $-$LLM planner", "M7": "M7 $-$Recomposition"}
VARS = list("NLHIPFRC")
RUNTIME = ["F", "R", "C"]


def load(name="runs"):
    rows = [json.loads(l) for l in open(RAW / f"{name}.jsonl")]
    err = [r for r in rows if "error" in r]
    if err:
        raise SystemExit(f"{len(err)} errored runs in {name}")
    return pd.DataFrame(rows)


def boot_ci(df, col, stat=np.mean):
    """Cluster bootstrap over instances (seeds averaged within instance)."""
    per = df.groupby("instance_id")[col].mean().dropna().values
    if len(per) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(per), (NBOOT, len(per)))
    bs = per[idx].mean(axis=1)
    return (per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def paired_boot(a, b):
    """a, b: per-instance means aligned; CI of mean difference."""
    d = a - b
    idx = RNG.integers(0, len(d), (NBOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def mcnemar(x, y):
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    orr = (b + 0.5) / (c + 0.5)
    return b, c, p, orr


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(max(0, min(1, p1)))) - 2 * math.asin(math.sqrt(max(0, min(1, p2))))


def fmt(m, lo=None, hi=None, d=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    if lo is None:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, s):
    (TAB / f"{name}.tex").write_text(s)


# ------------------------------------------------------------------ tables
def t1_main(df):
    lines = []
    summ = {}
    for m in BASE + ["P"]:
        d = df[df.method == m]
        cells = []
        rec = {}
        for col in ["TSR", "VWS", "SAA", "CVR"]:
            v = boot_ci(d, col)
            rec[col] = v
            cells.append(fmt(*v))
        v = boot_ci(d[d.variant.isin(RUNTIME)], "VWS")
        rec["SRR"] = v
        cells.append(fmt(*v))
        rec["PC"] = boot_ci(d, "PC")
        cells.append(fmt(rec["PC"][0]))
        rec["latency"] = boot_ci(d, "latency")
        rec["cost"] = boot_ci(d, "cost")
        cells.append(fmt(rec["latency"][0], d=1))
        cells.append(fmt(rec["cost"][0], d=3))
        summ[m] = {k: v[0] for k, v in rec.items()}
        nm = NAMES[m]
        if m == "P":
            nm = r"\textbf{ESAOG}"
        lines.append(nm + " & " + " & ".join(cells) + r"\\")
    body = "\n".join(lines)
    write("t1_main", rf"""\begin{{table}}[H]
\centering
\caption{{Overall results by method (SOST v1.0.1, SimLLM nominal; 192 SOST instances $\times$ 5 seeds = 960 paired runs per method). Mean with 95\% cluster-bootstrap CI over instances. SRR is VWS on runtime-perturbation variants (F, R, C). PFC = Required Provenance Field Coverage: share of the eight study-defined provenance fields present per execution decision (descriptive; systems with other native log schemas are structurally disadvantaged). Latency in seconds (simulated LLM + simulated tool + measured semantic overhead); cost in USD per workflow (nominal LLM pricing + declared tool cost).}}
\label{{tab:mainresults}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}}
\toprule
Method & TSR & VWS & SAA & CVR$\downarrow$ & SRR & PFC & Latency$\downarrow$ & Cost$\downarrow$\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    return summ


def t1b_robust(df):
    """Robustness of VWS to benchmark design choices: budget treated as soft; open vs templated goals;
    instances with vs without ontology modelling defects."""
    def nb(r):
        pol = sum(v for k, v in r["causes"].items() if k in ("ClearanceViolation", "ProhibitionViolation", "ObligationViolation"))
        return int(r["TSR"] == 1 and pol == 0)
    df = df.copy()
    df["VWS_nb"] = df.apply(nb, axis=1)
    lines, out = [], {}
    for m in BASE + ABL:
        d = df[df.method == m]
        r = dict(VWS=d.VWS.mean(), VWS_nb=d.VWS_nb.mean(), tmpl=d[d.template].VWS.mean(), open=d[~d.template].VWS.mean(),
                 nodef=d[~d.defect].VWS.mean(), defect=d[d.defect].VWS.mean())
        out[m] = r
        nm = r"\textbf{ESAOG}" if m == "P" else (ABL_NAMES[m] if m.startswith("M") else NAMES[m])
        lines.append(nm + " & " + " & ".join(fmt(r[k]) for k in ("VWS", "VWS_nb", "tmpl", "open", "nodef", "defect")) + r"\\")
    write("t1b_robust", r"""\begin{table}[H]
\centering
\caption{Sensitivity of VWS to benchmark design choices (SOST v1.0.1, SimLLM nominal). ``Budget soft'': VWS recomputed treating budget overruns as preferences (every instance contains one high-quality actor priced above the task budget). Templated (144 instances) vs.\ open goals (48 instances; no procedural template in $G_K$). Missing axiom (incomplete-ontology sensitivity): """ + str(df[df.defect].instance_id.nunique()) + r""" instances whose shipped ontology omits one subsumption/equivalence axiom.}
\label{tab:robust}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
Method & VWS & Budget soft & Templated & Open goal & Complete ont. & Missing axiom\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return out


def t2_domain(df):
    lines = []
    out = {}
    for m in BASE + ["P"]:
        cells = []
        for dom in ["literature", "software", "enterprise"]:
            d = df[(df.method == m) & (df.domain == dom)]
            v = d.VWS.mean()
            s = d[d.variant.isin(RUNTIME)].VWS.mean()
            out[(m, dom)] = (v, s)
            cells += [fmt(v), fmt(s)]
        nm = r"\textbf{ESAOG}" if m == "P" else NAMES[m]
        lines.append(nm + " & " + " & ".join(cells) + r"\\")
    write("t2_domain", r"""\begin{table}[H]
\centering
\caption{Domain-wise results (mean over 8 base workflows $\times$ 8 variants $\times$ 5 seeds per domain). VWS over all variants; SRR over F/R/C variants.}
\label{tab:domainresults}
\scriptsize
\begin{tabular}{lcccccc}
\toprule
Method & Lit.-VWS & Lit.-SRR & Soft.-VWS & Soft.-SRR & Ent.-VWS & Ent.-SRR\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return out


def t3_sost(df):
    lines = []
    piv = df[df.method.isin(BASE + ["P"])].groupby(["method", "variant"]).VWS.mean().unstack()[VARS]
    for m in BASE + ["P"]:
        row = piv.loc[m]
        cells = []
        for v in VARS:
            best = piv[v].max()
            s = fmt(row[v])
            cells.append(rf"\textbf{{{s}}}" if abs(row[v] - best) < 1e-9 else s)
        nm = r"\textbf{ESAOG}" if m == "P" else m
        lines.append(nm + " & " + " & ".join(cells) + r"\\")
    write("t3_sost", r"""\begin{table}[H]
\centering
\caption{SOST robustness: Valid Workflow Success by perturbation family (24 instances $\times$ 5 seeds per cell). Best value per column in bold (ties bolded).}
\label{tab:sostresults}
\scriptsize
\begin{tabular}{lcccccccc}
\toprule
Method & N & L & H & I & P & F & R & C\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return piv


def t4_ablation(df):
    lines = []
    out = {}
    ref = df[df.method == "P"]
    for m in ABL:
        d = df[df.method == m]
        vws, saa, cvr = d.VWS.mean(), d.SAA.mean(), d.CVR.mean()
        srr = d[d.variant.isin(RUNTIME)].VWS.mean()
        rr = d[d.variant.isin(RUNTIME)].RR.mean()
        lat = d.latency.mean()
        out[m] = dict(VWS=vws, SAA=saa, CVR=cvr, SRR=srr, RR=rr, latency=lat)
        lines.append(f"{ABL_NAMES[m]} & {fmt(vws)} & {fmt(saa)} & {fmt(cvr)} & {fmt(srr)} & {fmt(rr, d=3)} & {fmt(lat, d=1)}" + r"\\")
    piv = df[df.method.isin(ABL)].groupby(["method", "variant"]).VWS.mean().unstack()[VARS]
    l2 = []
    for m in ABL:
        l2.append(ABL_NAMES[m] + " & " + " & ".join(fmt(piv.loc[m][v]) for v in VARS) + r"\\")
    write("t4_ablation", r"""\begin{table}[H]
\centering
\caption{Ablation results (960 runs per variant). SRR and Recovery Regret (RR, oracle-utility units, lower is better) over F/R/C variants.}
\label{tab:ablationresults}
\small
\begin{tabular}{lcccccc}
\toprule
Variant & VWS & SAA & CVR$\downarrow$ & SRR & Recovery regret$\downarrow$ & Latency$\downarrow$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Ablation VWS by SOST family: which mechanism carries which perturbation.}
\label{tab:ablationsost}
\scriptsize
\begin{tabular}{lcccccccc}
\toprule
Variant & N & L & H & I & P & F & R & C\\
\midrule
""" + "\n".join(l2) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return out, piv


def t5_confusion(df):
    lines = []
    out = {}
    for m in BASE + ["P"] + ABL[1:]:
        d = df[df.method == m]
        f = d[d.runtime_failure == 1]
        nf = d[d.runtime_failure == 0]
        a = int(((f.VWS == 1)).sum())
        b = int(((f.TSR == 1) & (f.VWS == 0)).sum())
        c = int((f.TSR == 0).sum())
        e = int((nf.VWS == 1).sum())
        g = int((nf.VWS == 0).sum())
        cond = a / len(f) if len(f) else float("nan")
        out[m] = dict(fail_valid=a, fail_invalid_success=b, fail_unrecovered=c, nofail_valid=e, nofail_invalid=g, cond=cond)
        nm = r"\textbf{ESAOG}" if m == "P" else (ABL_NAMES.get(m, m) if m.startswith("M") else m)
        lines.append(f"{nm} & {len(f)} & {a} & {b} & {c} & {fmt(cond)} & {e} & {g}" + r"\\")
    write("t5_confusion", r"""\begin{table}[H]
\centering
\caption{Failure/recovery confusion matrix (counts over 960 runs per method). A run ``observes a failure'' when any tool invocation failed (injected unavailability, transient error, type mismatch, missing input, or output-governance rejection). Cond.\ SRR = recovered-and-valid / runs with an observed failure.}
\label{tab:confusion}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrcrr}
\toprule
& \multicolumn{5}{c}{Runtime failure observed} & \multicolumn{2}{c}{No failure observed}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Method & Runs & Recovered, valid & Succeeded, policy-invalid & Unrecovered & Cond.\ SRR & Valid & Invalid\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return out


CAUSE_COLS = [("capability:lexical_distractor", "Lexical distractor"), ("capability:sibling", "Sibling cap."),
              ("capability:overgeneral", "Over-general"), ("capability:generic", "Generic tool"),
              ("capability_other", "Other cap."), ("IOIncompatible", "I/O"), ("ClearanceViolation", "Clearance"),
              ("ProhibitionViolation", "Prohibition"), ("BudgetViolation", "Budget"), ("ObligationViolation", "Obligation"),
              ("unavailable_actor", "Unavail."), ("false_exclusion_no_candidate", "False excl.")]


def t6_causes(df):
    lines = []
    out = {}
    for m in BASE + ["P"] + ABL[1:]:
        d = df[df.method == m]
        tot = {}
        for c in d.causes:
            for k, v in c.items():
                key = k
                if k.startswith("capability:") and k not in [x for x, _ in CAUSE_COLS]:
                    key = "capability_other"
                tot[key] = tot.get(key, 0) + v
        out[m] = tot
        nm = r"\textbf{ESAOG}" if m == "P" else m
        lines.append(nm + " & " + " & ".join(str(tot.get(k, 0)) for k, _ in CAUSE_COLS) + r"\\")
    hdr = " & ".join(h for _, h in CAUSE_COLS)
    write("t6_causes", r"""\begin{table}[H]
\centering
\caption{Semantic assignment errors by cause (executed invocations summed over 960 runs per method; one invocation may have several causes). Capability errors are split by the distractor role that attracted the method. ``False excl.'' counts runs aborted because no eligible candidate remained (e.g.\ due to missing ontology axioms or missing entailment).}
\label{tab:causes}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "r" * len(CAUSE_COLS) + r"""}
\toprule
Method & """ + hdr + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return out


def t7_cost(df):
    lines = []
    out = {}
    for m in BASE + ["P"] + ABL[1:]:
        d = df[df.method == m]
        r = dict(lat_llm=d.lat_llm.mean(), lat_tool=d.lat_tool.mean(), t_reason=1000 * d.t_reasoning.mean(),
                 t_sparql=1000 * d.t_sparql.mean(), t_shacl=1000 * d.t_shacl.mean(), t_graph=1000 * d.t_graph.mean(),
                 tin=d.tokens_in.mean(), tout=d.tokens_out.mean(), calls=d.llm_calls.mean(), tools=d.tool_calls.mean(),
                 c_llm=d.cost_llm.mean(), c_tool=d.cost_tool.mean(), triples=d.graph_triples.mean())
        out[m] = r
        nm = r"\textbf{ESAOG}" if m == "P" else m
        lines.append(nm + f" & {r['lat_llm']:.1f} & {r['lat_tool']:.1f} & {r['t_reason']:.0f} & {r['t_sparql']:.0f} & "
                     f"{r['t_shacl']:.0f} & {r['t_graph']:.0f} & {r['tin']:.0f} & {r['tout']:.0f} & {r['calls']:.1f} & "
                     f"{r['tools']:.1f} & {r['c_llm']:.4f} & {r['c_tool']:.3f} & {r['triples']:.0f}" + r"\\")
    write("t7_cost", r"""\begin{table}[H]
\centering
\caption{Latency, token and cost decomposition (means per workflow run). LLM and tool latencies are simulated (s); reasoning (HermiT, warm cache), SPARQL, SHACL and graph-write times are measured wall-clock (ms) on the experiment machine. Cold HermiT classification is reported separately in the text. Triples = size of $G_K\cup G_C\cup G_W\cup G_E$ plus the materialised closure at the end of the run.}
\label{tab:costdecomp}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
Method & LLM s & Tool s & Reason ms & SPARQL ms & SHACL ms & Write ms & Tok.\ in & Tok.\ out & LLM calls & Tool calls & LLM \$ & Tool \$ & Triples\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return out


def per_instance(df, m, col, variants=None):
    d = df[df.method == m]
    if variants:
        d = d[d.variant.isin(variants)]
    return d.set_index(["instance_id", "seed"])[col].sort_index()


def inst_p(df, a, b, col="VWS", variants=None):
    """Cluster-level robustness test: Wilcoxon signed-rank on per-instance mean outcomes (seeds averaged)."""
    x, y = per_instance(df, a, col, variants), per_instance(df, b, col, variants)
    x, y = x.align(y, join="inner")
    xi, yi = x.groupby(level=0).mean(), y.groupby(level=0).mean()
    if ((xi - yi) != 0).sum() == 0:
        return 1.0
    return stats.wilcoxon(xi.values, yi.values).pvalue


def compare(df, a, b, col="VWS", variants=None):
    x, y = per_instance(df, a, col, variants), per_instance(df, b, col, variants)
    x, y = x.align(y, join="inner")
    xi = x.groupby(level=0).mean().values
    yi = y.groupby(level=0).mean().values
    diff, lo, hi = paired_boot(xi, yi)
    bb, cc, p, orr = mcnemar(x.values, y.values)
    return dict(diff=diff, lo=lo, hi=hi, p=p, b=bb, c=cc, OR=orr, h=cohen_h(x.mean(), y.mean()), pa=x.mean(), pb=y.mean(), n=len(x))


def interp(r, a="ESAOG", b="baseline"):
    if r["padj"] >= 0.05:
        return "no significant difference"
    return f"{a} higher" if r["diff"] > 0 else f"{a} lower"


def t8_stats(df, t1):
    comps = []
    for b in BASE:
        comps.append((f"ESAOG vs {b}", "VWS", compare(df, "P", b)))
    srr = {b: df[(df.method == b) & df.variant.isin(RUNTIME)].VWS.mean() for b in BASE}
    sb = max(srr, key=srr.get)
    comps.append((f"ESAOG vs strongest ({sb})", "SRR", compare(df, "P", sb, variants=RUNTIME)))
    fam_best = {}
    for fam in ["H", "I", "P", "F", "R"]:
        vv = {b: df[(df.method == b) & (df.variant == fam)].VWS.mean() for b in BASE}
        fb = max(vv, key=vv.get)
        fam_best[fam] = fb
        comps.append((f"ESAOG vs {fb} (SOST-{fam})", f"VWS-{fam}", compare(df, "P", fb, variants=[fam])))
    vwsb = max(BASE, key=lambda b: t1[b]["VWS"])
    adj = holm([c[2]["p"] for c in comps])
    for c, pa in zip(comps, adj):
        c[2]["padj"] = pa
    ivar = [None] * 9 + [RUNTIME] + [[f] for f in ["H", "I", "P", "F", "R"]]
    icmp = [(c[0].split("(")[1].rstrip(")") if "strongest" in c[0] else c[0].split(" vs ")[1].split(" ")[0]) for c in comps]
    ip = holm([inst_p(df, "P", b, variants=v) for b, v in zip(icmp, ivar)])
    for c, q in zip(comps, ip):
        c[2]["padj_inst"] = q
    lines = []
    for name, out, r in comps:
        lines.append(f"{name} & {out} & {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] & {r['b']}/{r['c']} & "
                     f"{r['h']:+.2f} & {fp(r['padj'])} & {fp(r['padj_inst'])} & {interp(r)}" + r"\\")
    # latency / cost Wilcoxon vs the strongest-VWS baseline
    lat = []
    for col in ["latency", "cost"]:
        x = per_instance(df, "P", col).groupby(level=0).mean()
        y = per_instance(df, vwsb, col).groupby(level=0).mean()
        x, y = x.align(y, join="inner")
        w = stats.wilcoxon(x.values, y.values)
        n = len(x)
        rb = 1 - 2 * w.statistic / (n * (n + 1) / 2)
        dd = paired_boot(x.values, y.values)
        lat.append((col, dd, w.pvalue, rb))
    ll = [f"ESAOG vs {vwsb} & {c} & {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] & -- & $r$={rb:+.2f} & -- & {fp(p)} & "
          f"{'no significant difference' if p >= 0.05 else ('ESAOG more expensive' if d[0] > 0 else 'ESAOG cheaper')}" + r"\\" for c, d, p, rb in lat]
    write("t8_stats", r"""\begin{table}[H]
\centering
\caption{Paired statistical comparisons (SOST v1.0.1, SimLLM nominal). Effect = difference in proportion (ESAOG $-$ comparator) with 95\% cluster-bootstrap CI over instances; b/c = discordant instance$\times$seed pairs (ESAOG-only valid / comparator-only valid); $h$ = Cohen's $h$. Run-level $p$: exact McNemar over the 960 instance$\times$seed pairs, Holm-adjusted across the 15 binary comparisons; because the five seeds of an instance are not independent, these $p$-values are anti-conservative. Instance-level $p$: Wilcoxon signed-rank on per-instance mean outcomes (192 instances; 72 for SRR; 24 per SOST family), Holm-adjusted; this is the inferentially appropriate test. Continuous outcomes: Wilcoxon signed-rank on instance means, rank-biserial $r$. SOST-family comparators are the strongest baseline in that family.}
\label{tab:statsresults}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccll}
\toprule
Comparison & Outcome & Effect (95\% CI) & b/c & Effect size & Run-level adj.\ $p$ & Instance-level adj.\ $p$ & Interpretation \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
""" + "\n".join(ll) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # ablations vs full
    ab = [(m, compare(df, m, "P")) for m in ABL[1:]]
    adj = holm([r["p"] for _, r in ab])
    l3 = []
    ipa = holm([inst_p(df, m, "P") for m, _ in ab])
    for (m, r), pa, q in zip(ab, adj, ipa):
        r["padj"] = pa
        r["padj_inst"] = q
        l3.append(f"{ABL_NAMES[m]} vs M0 & VWS & {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] & {r['b']}/{r['c']} & "
                  f"{r['h']:+.2f} & {fp(pa)} & {fp(q)}" + r"\\")
    write("t8b_ablstats", r"""\begin{table}[H]
\centering
\caption{Ablation contrasts against the full model (SOST v1.0.1, SimLLM nominal). Effect with 95\% cluster-bootstrap CI over instances. Run-level $p$: exact McNemar over instance$\times$seed pairs (anti-conservative, see Table~\ref{tab:statsresults}); instance-level $p$: Wilcoxon signed-rank on per-instance means (192 instances). Both Holm-adjusted across the 7 ablations. Negative effect = the removed mechanism contributed to VWS.}
\label{tab:ablstats}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
Contrast & Outcome & Effect (95\% CI) & b/c & $h$ & Run-level adj.\ $p$ & Instance-level adj.\ $p$\\
\midrule
""" + "\n".join(l3) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return comps, lat, ab, sb, fam_best, vwsb


def t9_coverage():
    f = RAW / "coverage.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    o = c["ontology"]
    rows = []
    for d in ["literature", "software", "enterprise"]:
        x = o[d]
        rows.append(f"{d.capitalize()} & {x['capabilities']} & {x['data_types']} & {x['subclass_axioms']} & "
                    f"{x['equivalence_axioms']} & {x['max_capability_depth']} & {x['slots']} & {x['triples']}" + r"\\")
    det = []
    for k, v in sorted(c["detection"].items()):
        det.append(f"{k} & {v['expected']} & {v['detected']} & {v['rate']:.3f}" + r"\\")
    fe = c["false_exclusions"]
    sh = o["shapes"]
    write("t9_coverage", r"""\begin{table}[H]
\centering
\caption{Ontology and SHACL coverage. Top: domain extensions of the shared upper ontology (core: """ +
          f"{o['core']['classes']} classes, {o['core']['object_properties']} object and {o['core']['datatype_properties']} datatype properties" +
          r"""). Middle: workflow/registry shapes and SPARQL queries. Bottom: evaluator unit tests---every expected check declared in the benchmark instances, and whether ESAOG's gates (A1 SPARQL eligibility or SHACL) flag it when the offending actor is placed in an otherwise valid workflow.}
\label{tab:coverage}
\scriptsize
\begin{tabular}{lrrrrrrr}
\toprule
Domain & Capabilities & Data types & subClassOf & equivalentClass & Max depth & Task slots & Triples\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lr}
\toprule
Workflow node shapes / SPARQL constraints / property constraints & """ +
          f"{sh['workflow_node_shapes']} / {sh['workflow_sparql_constraints']} / {sh['workflow_property_constraints']}" + r"""\\
Registry (A4) node shapes / property constraints & """ + f"{sh['registry_node_shapes']} / {sh['registry_property_constraints']}" + r"""\\
SPARQL query files & """ + f"{sh['sparql_queries']}" + r"""\\
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lrrr}
\toprule
Expected check & Expected & Detected & Rate\\
\midrule
""" + "\n".join(det) + r"""
\midrule
\multicolumn{4}{l}{Valid alternatives tested: """ + f"{c['valid_alternatives_tested']}; falsely excluded: {fe.get('defect', 0)} in missing-axiom instances, {fe.get('no_defect', 0)} otherwise" + r"""}\\
\bottomrule
\end{tabular}
\end{table}
""")
    return c


FINAL_RAW = ["runs.jsonl", "sens_weak.jsonl", "sens_strong.jsonl", "coverage.json",
             "real_gpt-6-luna_all.jsonl", "real_gpt-6-luna_all_v11.jsonl", "real_claude-haiku-4-5_all.jsonl",
             "real_llama3.2-3b_sub.jsonl", "real_llama3.2-3b_sub_v11.jsonl", "real_llama3.2-3b_rest.jsonl", "real_qwen3-4b_sub.jsonl",
             "real_qwen3-4b_sub_v11.jsonl", "real_qwen3-4b_rest.jsonl", "real_llama3.1-8b_sub.jsonl",
             "real_llama3.1-8b_rest.jsonl", "real_qwen2.5-14b_sub.jsonl", "real_qwen2.5-14b_rest.jsonl"]


def sha_files(paths):
    import hashlib
    h = hashlib.sha256()
    for f in paths:
        h.update(Path(f).name.encode()); h.update(Path(f).read_bytes())
    return h.hexdigest()


def t10_manifest():
    import hashlib
    m = json.loads((RAW / "manifest_runs.json").read_text())
    p = m["packages"]
    bench = ROOT / "benchmark" / "sost_benchmark.json"
    bsha = hashlib.sha256(bench.read_bytes()).hexdigest()
    onto = sorted((ROOT / "ontology").glob("*.ttl")) + sorted((ROOT / "shapes").glob("*.ttl"))
    raw = [RAW / f for f in FINAL_RAW if (RAW / f).exists()]
    ana = [ROOT / "scripts" / f for f in ("analysis.py", "analysis_real.py", "sensitivity.py", "coverage.py", "fill_paper.py")]
    AA = r"[AUTHOR ACTION REQUIRED: %s]"
    shacl = sorted((ROOT / "shapes").glob("*.ttl"))
    ontf = sorted((ROOT / "ontology").glob("*.ttl"))
    runners = [ROOT / "run_all.sh"] + sorted((ROOT / "scripts").glob("run_*.sh")) + [ROOT / "scripts" / "run_experiments.py", ROOT / "scripts" / "qwen14b_rest.sh"]
    asrun = RAW / "sost_benchmark_v1.0.1_as_run.json"
    rows = [("ESAOG architecture version (final)", "ESAOG v1.1 (post-observation engineering revision; dataflow grounding)"),
            ("Original evaluated architecture", "ESAOG v1.0 (pre-specified; reported alongside v1.1 for every real model)"),
            ("Post-observation engineering revision", "ESAOG v1.1: dataflow edges derived from typed task signatures in $G_K$, introduced after the first real-model run"),
            ("Software/repository release", AA % "tag a release"),
            ("Git commit", AA % "insert commit hash"),
            ("Repository URL", AA % "insert repository URL"),
            ("Zenodo DOI", AA % "insert Zenodo DOI after creating final release"),
            ("SOST final version", f"v1.0.1, {m['benchmark']['n_instances']} instances (\\texttt{{benchmark/sost\\_benchmark.json}})"),
            ("SOST final SHA-256", bsha),
            ("SOST as-run file SHA-256", hashlib.sha256(asrun.read_bytes()).hexdigest() + " (identical content; metadata version label read 1.0.0 when the runs were made; this is the hash recorded in the run manifests)"),
            ("SOST superseded version / SHA-256", "v1.0.0 (before the post-protocol wording correction; kept only for Table~\\ref{tab:benchver}): " + hashlib.sha256((RAW / "sost_benchmark_v1.0.0_superseded.json").read_bytes()).hexdigest()),
            ("Ontology version / SHA-256", f"{m['ontology_version']} / " + sha_files(ontf) + f" ({len(ontf)} TTL files)"),
            ("SHACL shapes SHA-256", sha_files(shacl) + f" ({len(shacl)} TTL files)"),
            ("Raw results SHA-256", sha_files(raw) + f" ({len(raw)} files in results/raw)"),
            ("Analysis scripts SHA-256", sha_files(ana) + " (analysis.py, analysis\\_real.py, sensitivity.py, coverage.py, fill\\_paper.py)"),
            ("Literature evidence matrix SHA-256", sha_files([ROOT / "docs" / "literature_evidence_matrix.csv"]) + " (docs/literature\\_evidence\\_matrix.csv)"),
            ("Experiment runner scripts SHA-256", sha_files(runners) + f" ({len(runners)} files; run parameters are the command lines in these scripts)"),
            ("Code tree hash, main run (SHA-256/16)", m["code_tree_sha256_16"] + " (computed at run completion; per-run hashes in results/raw/manifest\\_*.json)"),
            ("Master seed / run seeds", f"{m['benchmark']['master_seed']} / {','.join(map(str, m['seeds']))} (SimLLM); one run per instance for real LLMs"),
            ("Main runs", f"{m['n_runs']} ({len(m['methods'])} configurations), {m['errors']} errors"),
            ("SimLLM", f"profile {m['llm']['profile']} (weak/strong for sensitivity); prompts {m['llm']['prompts']}; pricing assumption USD 3/M input, 15/M output tokens"),
            ("Hosted LLM 1", "OpenAI gpt-6-luna, Chat Completions, JSON mode, seed 0, reasoning\\_effort=low, default sampling (temperature not accepted); run 2026-09-27; USD 0.10/0.50 per M tokens"),
            ("Hosted LLM 2", "Anthropic claude-haiku-4-5, Messages API, temperature 0, max 4{,}000 output tokens, no extended thinking; run 2026-09-27; USD 1/5 per M tokens"),
            ("Local LLMs (Ollama)", "Llama-3.2-3B-Instruct, Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct (Q4\\_K\\_M GGUF, Hugging Face), qwen2.5:14b (Q4\\_K\\_M); temperature 0, seed 0, JSON format; context 8{,}192 (4{,}096 for 8B/14B); runs 2026-09-27 to 2026-09-29"),
            ("Reasoner", f"HermiT via owlready2 {p['owlready2']} ({p['java']})"),
            ("RDF / SPARQL / SHACL", f"rdflib {p['rdflib']}, pySHACL {p['pyshacl']}"),
            ("Runtime adapters", f"LangGraph {p['langgraph']} StateGraph adapter and a direct sequential adapter (both in this release); MCP-style JSON-RPC tool server (not a conformance-tested MCP implementation)"),
            ("Statistics", f"numpy {p['numpy']}, scipy {p['scipy']}, pandas {p['pandas']}"),
            ("Python / platform", f"{m['python']} / {m['platform']}"),
            ("Cache policy", m["cache_policy"].replace("no caching of LLM calls", "no caching of SimLLM calls") + "; real-LLM completions cached by exact prompt (identical prompts issued by different methods receive the identical completion)"),
            ("One-command runner", r"\texttt{bash run\_all.sh}")]
    esc = lambda b: b if (b.startswith("\\texttt") or "\\_" in b or "AUTHOR ACTION" in b or "\\ref" in b or "$" in b) else str(b).replace("_", chr(92) + "_")
    body = "\n".join(f"{a} & {esc(b)}" + r"\\" for a, b in rows)
    write("t10_manifest", r"""\begin{table}[H]
\centering
\caption{Reproducibility manifest. Hashes are computed from the released files at table-generation time; per-run manifests are in \texttt{results/raw/manifest\_*.json}.}
\label{tab:manifest}
\scriptsize
\begin{tabularx}{\textwidth}{p{3.6cm}X}
\toprule
Item & Value\\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    return m


# ------------------------------------------------------------------ figures
COL = {"P": "#c0392b"}


def fig_pareto(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, (x, y, lab) in enumerate([("cost", "VWS", "Valid Workflow Success"), ("latency", "SRR", "Semantic Recovery Rate (F/R/C)")]):
        pts = []
        for m in BASE + ["P"]:
            d = df[df.method == m]
            xv = d[x].mean()
            yv = d[d.variant.isin(RUNTIME)].VWS.mean() if y == "SRR" else d.VWS.mean()
            pts.append((xv, yv, m))
        front = []
        for p in sorted(pts):
            if not front or p[1] > front[-1][1]:
                front.append(p)
        ax[k].plot([p[0] for p in front], [p[1] for p in front], "--", color="grey", lw=1, label="Pareto front")
        for xv, yv, m in pts:
            ax[k].scatter(xv, yv, s=60 if m == "P" else 35, color=COL.get(m, "#2c3e50"), zorder=3)
            ax[k].annotate("ESAOG" if m == "P" else m, (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax[k].set_xlabel("Mean cost per workflow (USD)" if x == "cost" else "Mean end-to-end latency (s)")
        ax[k].set_ylabel(lab)
        ax[k].grid(alpha=.3)
        ax[k].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "pareto.pdf")
    plt.close(fig)


def fig_perturbation(piv):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ms = BASE + ["P"]
    w = 0.08
    for i, m in enumerate(ms):
        ax.bar(np.arange(len(VARS)) + (i - len(ms) / 2) * w, piv.loc[m].values, w, label="ESAOG" if m == "P" else m,
               color=COL.get(m, plt.cm.tab20(i / 10)))
    ax.set_xticks(np.arange(len(VARS)))
    ax.set_xticklabels([f"SOST-{v}" for v in VARS])
    ax.set_ylabel("VWS")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=10, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "per_perturbation.pdf")
    plt.close(fig)


def fig_ablation(df, ab):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = [ABL_NAMES[m].replace("$-$", "−") for m, _ in ab]
    d = [r["diff"] for _, r in ab]
    lo = [r["diff"] - r["lo"] for _, r in ab]
    hi = [r["hi"] - r["diff"] for _, r in ab]
    ax.barh(names, d, xerr=[lo, hi], color=["#c0392b" if x < 0 else "#27ae60" for x in d], capsize=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("ΔVWS vs full ESAOG (95% cluster-bootstrap CI)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_effects.pdf")
    plt.close(fig)


def fig_hierarchy():
    sys.path.insert(0, str(ROOT / "src"))
    from esaog.domains import cap_index, DOMAINS
    parent, children, label, _, _ = cap_index("literature")
    eq = DOMAINS["literature"]["equiv"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    pos = {"QualityAppraisal": (3.6, 3), "RiskOfBiasAssessment": (2.2, 2), "ReviewQualityAppraisal": (5.5, 2),
           "RoB2Assessment": (1.2, 1), "ROBINSIAssessment": (3.2, 1), "AMSTAR2Appraisal": (5.5, 1),
           "CochraneRiskOfBias2": (1.2, 0), "LiteratureSearch": (9.2, 3), "BiomedicalSearch": (8.4, 2),
           "PubMedSearch": (8.4, 1), "PreprintSearch": (10.7, 2), "WebSearch": (11.6, 3)}
    for c, (x, y) in pos.items():
        p = parent.get(c)
        if p in pos:
            ax.annotate("", xy=pos[p], xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="#2c3e50"))
    for a, b in eq:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[a], xytext=pos[b], arrowprops=dict(arrowstyle="<|-|>", ls="--", color="#c0392b"))
            ax.text((pos[a][0] + pos[b][0]) / 2 + .1, (pos[a][1] + pos[b][1]) / 2, "owl:equivalentClass", color="#c0392b", fontsize=7)
    for c, (x, y) in pos.items():
        ax.text(x, y, label[c], ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="#fdebd0" if c == "WebSearch" else "#eaf2f8", ec="#2c3e50"))
    ax.text(0.2, 3.3, "rdfs:subClassOf (solid)", fontsize=7)
    ax.set_xlim(-0.2, 12.4)
    ax.set_ylim(-0.5, 3.6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "capability_hierarchy.pdf")
    plt.close(fig)


def main():
    df = load("runs")
    t1 = t1_main(df)
    t1b = t1b_robust(df)
    t2 = t2_domain(df)
    piv = t3_sost(df)
    t4, apiv = t4_ablation(df)
    t5 = t5_confusion(df)
    t6 = t6_causes(df)
    t7 = t7_cost(df)
    comps, lat, ab, sb, fam_best, vwsb = t8_stats(df, t1)
    t9 = t9_coverage()
    t10 = t10_manifest()
    fig_pareto(df)
    fig_perturbation(piv)
    fig_ablation(df, ab)
    fig_hierarchy()
    cold = pd.Series([json.loads(f.read_text())["cold_s"] for f in (ROOT / "results" / "cache").glob("hermit_*.json")])
    summary = dict(t1=t1, robust=t1b, sost={m: piv.loc[m].to_dict() for m in piv.index}, ablation=t4,
                   ablation_sost={m: apiv.loc[m].to_dict() for m in apiv.index}, confusion=t5, causes=t6, cost=t7,
                   stats=[(n, o, r) for n, o, r in comps], lat=[(c, d, p, rb) for c, d, p, rb in lat],
                   abl_stats=[(m, r) for m, r in ab], strongest_srr=sb, fam_best=fam_best, strongest_vws=vwsb,
                   hermit_cold_mean_s=float(cold.mean()) if len(cold) else None,
                   by_template={m: df[df.method == m].groupby("template").VWS.mean().to_dict() for m in BASE + ABL},
                   by_defect={m: df[df.method == m].groupby("defect").VWS.mean().to_dict() for m in BASE + ABL},
                   domain={f"{m}|{d}": v for (m, d), v in t2.items()})
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(dict(t1=t1, strongest_srr=sb, fam_best=fam_best, strongest_vws=vwsb), indent=1, default=float))


if __name__ == "__main__":
    main()
