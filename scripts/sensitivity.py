"""SimLLM competence sensitivity (weak / nominal / strong) for LLM-dependent methods (seeds 0,1)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
M = ["B1", "B4", "B6", "B8", "P", "M6"]
NM = {"B1": "B1 LLM-only", "B4": "B4 Tressoir-like", "B6": "B6 O-MCP-like", "B8": "B8 COVENANT-like", "P": "ESAOG", "M6": "M6 $-$LLM planner"}


def load(n):
    return pd.DataFrame([json.loads(l) for l in open(RAW / f"{n}.jsonl")])


def main():
    nom = load("runs")
    nom = nom[nom.method.isin(M) & nom.seed.isin([0, 1])]
    frames = {"weak": load("sens_weak"), "nominal": nom, "strong": load("sens_strong")}
    out, lines = {}, []
    for m in M:
        cells = []
        for p in ["weak", "nominal", "strong"]:
            d = frames[p][frames[p].method == m]
            v, s = d.VWS.mean(), d[d.variant.isin(["F", "R", "C"])].VWS.mean()
            hi = d[d.variant.isin(list("HIPFR"))].VWS.mean()
            out[f"{m}|{p}"] = dict(VWS=v, SRR=s, HIPFR=hi, tokens=d.tokens_in.mean() + d.tokens_out.mean())
            cells += [f"{v:.2f}", f"{hi:.2f}", f"{s:.2f}"]
        lines.append(NM[m] + " & " + " & ".join(cells) + r"\\")
    (ROOT / "results" / "tables" / "t11_sensitivity.tex").write_text(r"""\begin{table}[H]
\centering
\caption{Sensitivity of the conclusions to the simulated LLM's competence (SimLLM profiles; 192 instances $\times$ seeds 0--1 = 384 runs per cell). H--R = VWS on SOST-H/I/P/F/R; SRR on F/R/C. M6 does not call an LLM and serves as a control.}
\label{tab:sensitivity}
\scriptsize
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Weak} & \multicolumn{3}{c}{Nominal} & \multicolumn{3}{c}{Strong}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Method & VWS & H--R & SRR & VWS & H--R & SRR & VWS & H--R & SRR\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (ROOT / "results" / "sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
