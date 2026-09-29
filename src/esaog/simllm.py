"""SimLLM: a seeded, parametric stand-in for an LLM proposal generator.

DEVIATION (documented in the paper): no LLM API credentials were available to the
experiment environment, so every LLM call is simulated. SimLLM is a *behavioural* model:
  * it reads the same natural-language text an LLM would (goal, task and actor
    descriptions; metadata/history/feedback only when the method puts them in the prompt),
  * it scores candidates by lexical similarity (TF-IDF), a latent "world-knowledge"
    judgement of capability fit whose accuracy decays with ontological distance,
    stochastic attention to stated constraints, declared quality/cost, and Gumbel noise,
  * token counts are computed from the actual filled prompt strings (chars/4).
Latent judgements are keyed by (seed, instance, slot, actor) and NOT by method, so all
LLM-based methods share common random numbers (paired design). Competence profiles
(weak/nominal/strong) are provided for sensitivity analysis.
A real-LLM backend can be plugged in through the same interface (see llm_backend.py).
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from pathlib import Path

from . import oracle
from .domains import cap_index

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = {p.stem: p.read_text() for p in (ROOT / "prompts").glob("*.txt")}
STOP = set("a an the of for to and on with in by from into is are be that can its their this any all per as or at".split())

PROFILES = {
    "nominal": dict(p_direct=0.95, p_decay=0.80, p_equiv=0.55, p_overgeneral=0.40, p_sibling=0.30, p_unrelated=0.05,
                    p_attend=0.60, p_attend_policy=0.55, p_false_alarm=0.15, p_history=0.60, p_follow=0.95,
                    p_omit=0.03, p_wrong=0.04, noise=0.30),
    "weak": dict(p_direct=0.90, p_decay=0.65, p_equiv=0.35, p_overgeneral=0.55, p_sibling=0.45, p_unrelated=0.10,
                 p_attend=0.40, p_attend_policy=0.35, p_false_alarm=0.25, p_history=0.40, p_follow=0.85,
                 p_omit=0.06, p_wrong=0.08, noise=0.45),
    "strong": dict(p_direct=0.99, p_decay=0.92, p_equiv=0.80, p_overgeneral=0.20, p_sibling=0.12, p_unrelated=0.02,
                   p_attend=0.85, p_attend_policy=0.85, p_false_alarm=0.05, p_history=0.85, p_follow=0.99,
                   p_omit=0.01, p_wrong=0.015, noise=0.20),
}
PRICE_IN, PRICE_OUT = 3.0e-6, 15.0e-6  # USD/token, nominal frontier-model tier (assumption)


def krng(*keys) -> random.Random:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def tokenize(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


class TfIdf:
    def __init__(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(tokenize(d)))
        self.n = len(docs)
        self.idf = {w: math.log((1 + self.n) / (1 + c)) + 1 for w, c in df.items()}

    def vec(self, s):
        tf = Counter(tokenize(s))
        v = {w: c * self.idf.get(w, math.log(1 + self.n) + 1) for w, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}

    def cos(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return sum(x * vb.get(w, 0.0) for w, x in va.items())


class Meter:
    def __init__(self):
        self.tokens_in = self.tokens_out = self.calls = 0
        self.latency = 0.0

    def charge(self, prompt: str, out_tokens: int):
        tin = max(1, len(prompt) // 4)
        self.tokens_in += tin
        self.tokens_out += out_tokens
        self.calls += 1
        self.latency += 0.4 + 0.015 * out_tokens + 0.0002 * tin

    @property
    def cost(self):
        return self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT


class SimLLM:
    def __init__(self, inst, seed, profile="nominal", meter: Meter | None = None):
        self.inst, self.seed = inst, seed
        self.p = PROFILES[profile]
        self.meter = meter or Meter()
        self.dom = inst["domain"]
        self.parent, self.children, self.label, self.alt, _ = cap_index(self.dom)
        docs = [a["description"] for a in inst["available_actors"]] + [t["text"] for t in inst["required_tasks"]]
        self.tfidf = TfIdf(docs)
        self.hist = {h["actor"]: h for h in inst["provenance_history"]}

    # ------------------------------------------------------------ decomposition
    def decompose(self, feedback="", key="dec"):
        tasks = self.inst["required_tasks"]
        prompt = PROMPTS["decompose"].format(goal=self.inst["goal"]["text"], context=self.dom, feedback=feedback)
        planned = []
        for t in tasks:
            r = krng(self.seed, self.inst["instance_id"], "decompose", key, t["slot"])
            u = r.random()
            if u < self.p["p_omit"]:
                continue
            believed = t["required_capability"]
            if u < self.p["p_omit"] + self.p["p_wrong"]:
                believed = self._wrong_label(believed, r)
            planned.append(dict(slot=t["slot"], believed=believed, text=t["text"], task=t,
                                inputs=[i for i in t["inputs"]]))
        self.meter.charge(prompt, 60 * len(planned))
        kept = {p["slot"] for p in planned}
        for p in planned:
            p["consumes"] = [i for i in p["inputs"] if i != "goal" and i in kept]
            p["goal_input"] = p["task"]["in_types"][0] if "goal" in p["inputs"] else None
        return planned

    def _wrong_label(self, req, r):
        opts = []
        p = self.parent.get(req)
        if p:
            opts.append(p)
            opts += [s for s in self.children.get(p, []) if s != req]
        opts += self.children.get(req, [])
        return r.choice(opts) if opts else req

    def repair_decomposition(self, planned, missing_slots, key):
        """Planner receives MissingProducer violations and re-inserts the missing tasks."""
        prompt = PROMPTS["decompose"].format(goal=self.inst["goal"]["text"], context=self.dom,
                                             feedback="VIOLATIONS: missing producers for " + ", ".join(missing_slots))
        byslot = {t["slot"]: t for t in self.inst["required_tasks"]}
        have = {p["slot"] for p in planned}
        for s in missing_slots:
            if s in have or s not in byslot:
                continue
            if krng(self.seed, self.inst["instance_id"], "repair_dec", key, s).random() < self.p["p_follow"]:
                t = byslot[s]
                planned.append(dict(slot=s, believed=t["required_capability"], text=t["text"], task=t, inputs=list(t["inputs"])))
        order = [t["slot"] for t in self.inst["required_tasks"]]
        planned.sort(key=lambda p: order.index(p["slot"]))
        kept = {p["slot"] for p in planned}
        for p in planned:
            p["consumes"] = [i for i in p["inputs"] if i != "goal" and i in kept]
            p["goal_input"] = p["task"]["in_types"][0] if "goal" in p["inputs"] else None
        self.meter.charge(prompt, 60 * len(planned))
        return planned

    # ------------------------------------------------------------ assignment
    def belief_valid(self, actor, believed_req, slot):
        """Latent LLM judgement 'actor can perform believed_req' (stable within a seed)."""
        r = krng(self.seed, self.inst["instance_id"], "belief", slot, actor["uri"], believed_req).random()
        caps = actor["capabilities"]
        best = 0.0
        for c in caps:
            if c == believed_req:
                p = self.p["p_direct"]
            elif oracle.subsumed(self.dom, c, believed_req) and oracle.subsumed(self.dom, believed_req, c):
                p = self.p["p_equiv"]
            elif oracle.subsumed(self.dom, c, believed_req):
                d, x = 0, c
                while x and x != believed_req:
                    x, d = self.parent.get(x), d + 1
                d = d if x == believed_req else 2
                p = self.p["p_decay"] ** d
            elif oracle.subsumed(self.dom, believed_req, c):
                p = self.p["p_overgeneral"]
            elif self.parent.get(c) and self.parent.get(c) == self.parent.get(believed_req):
                p = self.p["p_sibling"]
            else:
                p = self.p["p_unrelated"]
            best = max(best, p)
        return r < best

    def choose(self, planned, candidates, upstream_types, info=("meta",), excluded=(), key="assign", attempt=0):
        """Return (actor_uri or None, justification)."""
        if not candidates:
            return None, "no candidates"
        slot, believed, text = planned["slot"], planned["believed"], planned["text"]
        task = planned["task"]
        lines = []
        for a in candidates:
            ln = f"- {a['uri']}: {a['description']}"
            if "meta" in info:
                ln += (f" [capability={','.join(a['capabilities'])}; input={','.join(a['accepts'])}; output={a['produces']};"
                       f" clearance={a['clearance']}; external={a['external']}; kind={a['kind']}; cost={a['cost']}]")
            lines.append(ln)
        hist = ""
        if "history" in info:
            hist = "HISTORY:\n" + "\n".join(f"{h['actor']}: {h['successes']}/{h['executions']} ok" for h in
                                             (self.hist.get(a["uri"]) for a in candidates) if h)
        meta = (f"required input types: {upstream_types}; required output: {task['out_type']}; sensitivity: "
                f"{task['sensitivity']}; personal data: {task['pii']}; budget: {task['max_cost']}") if "meta" in info else ""
        prompt = PROMPTS["assign"].format(goal=self.inst["goal"]["text"], task=text, task_meta=meta,
                                          candidates="\n".join(lines), history=hist,
                                          feedback=("EXCLUDE: " + ", ".join(excluded)) if excluded else "")
        self.meter.charge(prompt, 45)
        rn = krng(self.seed, self.inst["instance_id"], "noise", key, slot, attempt)
        best, best_s = None, -1e9
        for a in candidates:
            if a["uri"] in excluded and krng(self.seed, self.inst["instance_id"], "follow", key, slot, a["uri"], attempt).random() < self.p["p_follow"]:
                continue
            ra = krng(self.seed, self.inst["instance_id"], "attend", slot, a["uri"])
            s = 1.0 * self.tfidf.cos(text, a["description"])
            s += 1.2 if self.belief_valid(a, believed, slot) else -1.2
            if "meta" in info:
                io_bad = not (oracle.input_ok(self.dom, a, upstream_types) and oracle.output_ok(self.dom, a, task["out_type"]))
                exact = all(t in a["accepts"] for t in upstream_types) and a["produces"] == task["out_type"]
                if io_bad and ra.random() < self.p["p_attend"]:
                    s -= 1.5
                elif not io_bad and not exact and ra.random() < self.p["p_false_alarm"]:
                    s -= 1.5
                if oracle.policy_violations(self.dom, a, task) and ra.random() < self.p["p_attend_policy"]:
                    s -= 1.5
            if "history" in info:
                h = self.hist.get(a["uri"])
                if h and h["successes"] / h["executions"] < 0.6 and ra.random() < self.p["p_history"]:
                    s -= 1.0
            s += 0.5 * a["quality"] - 0.3 * min(a["cost"] / 0.1, 1.0)
            s += self.p["noise"] * -math.log(-math.log(max(1e-12, rn.random())))
            if s > best_s:
                best, best_s = a, s
        if best is None:
            return None, "all candidates excluded"
        return best["uri"], f"LLM choice for {believed}: score {best_s:.2f}"
