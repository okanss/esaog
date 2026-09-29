"""RealLLM: drop-in replacement for SimLLM that prompts an actual model (Claude, OpenAI, Ollama).

Same interface as SimLLM (decompose, repair_decomposition, choose, meter, tfidf, label, parent,
children), so every method runs unchanged. Differences from SimLLM that matter for interpretation:
  * decomposition: the model receives the goal, a prose brief of what the workflow must achieve, and
    the domain capability vocabulary; it returns tasks with a capability label and dependencies.
    Returned tasks are aligned to benchmark task slots by TF-IDF similarity of their descriptions
    (one-to-one, threshold 0.15); unaligned gold tasks count as omitted, unaligned extra tasks are dropped.
    Capability labels are grounded by exact label/altLabel/name match, else nearest label (logged).
  * dependencies come from the model's `depends_on` (a wrong edge produces a real MissingInput).
  * every judgement (capability fit, constraint attention, reliability) is the model's own.
Modelled overhead tokens that methods charge without issuing a call (B4 blueprint, B7 node/governance
messages, B8 compile) are still charged through `meter.charge` and priced at the provider's rates.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from .domains import DOMAINS, cap_index
from .providers import Provider
from .simllm import PROMPTS, TfIdf, tokenize

SYSTEM_PLAN = ("You are the planning component of a multi-agent workflow orchestrator. "
               "Answer with a single JSON object and nothing else.")
SYSTEM_ASSIGN = ("You are the orchestration component that assigns each workflow task to one actor (agent or tool). "
                 "Answer with a single JSON object and nothing else.")


class RealMeter:
    """Same fields as simllm.Meter; tokens/latency are the provider's reported values."""

    def __init__(self, provider: Provider):
        self.p = provider
        self.tokens_in = self.tokens_out = self.calls = 0
        self.latency = 0.0
        self.real_calls = self.cache_hits = 0
        self.modelled_in = self.modelled_out = 0

    def record(self, c):
        self.tokens_in += c.tokens_in
        self.tokens_out += c.tokens_out
        self.calls += 1
        self.latency += c.latency
        self.cache_hits += int(c.cached)
        self.real_calls += int(not c.cached)

    def charge(self, prompt: str, out_tokens: int):  # modelled overhead (no call issued)
        tin = max(1, len(prompt) // 4)
        self.tokens_in += tin
        self.tokens_out += out_tokens
        self.modelled_in += tin
        self.modelled_out += out_tokens
        self.calls += 1
        self.latency += 0.4 + 0.015 * out_tokens + 0.0002 * tin

    @property
    def cost(self):
        pi, po = self.p.price
        return self.tokens_in * pi + self.tokens_out * po


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


class RealLLM:
    def __init__(self, inst, seed, provider: Provider, meter=None):
        self.inst, self.seed, self.prov = inst, seed, provider
        self.meter = meter or RealMeter(provider)
        self.dom = inst["domain"]
        self.parent, self.children, self.label, self.alt, self.gloss = cap_index(self.dom)
        docs = [a["description"] for a in inst["available_actors"]] + [t["text"] for t in inst["required_tasks"]]
        self.tfidf = TfIdf(docs)
        self.hist = {h["actor"]: h for h in inst["provenance_history"]}
        self.stats = Counter()
        self._lab2cap = {}
        for c in self.label:
            for l in [self.label[c], c] + list(self.alt.get(c, [])):
                self._lab2cap[l.lower().strip()] = c
        self._labdocs = TfIdf(list(self.label.values()))

    # ------------------------------------------------------------ helpers
    def _ask(self, system, user, max_tokens=1024):
        c = self.prov.complete(system, user, max_tokens)
        self.meter.record(c)
        if c.stop == "refusal":
            self.stats["refusals"] += 1
        return parse_json(c.text)

    def ground(self, label):
        if not label:
            return None
        k = str(label).lower().strip()
        if k in self._lab2cap:
            return self._lab2cap[k]
        self.stats["label_fuzzy"] += 1
        best = max(self.label, key=lambda c: self._labdocs.cos(k, self.label[c]))
        return best

    def brief(self):
        # typed I/O of each required step is part of the task description in G_K and is given to every method
        return " ".join(f"({i + 1}) {t['text'][0].upper() + t['text'][1:]} [consumes: {', '.join(t['in_types'])}; "
                        f"produces: {t['out_type']}]." for i, t in enumerate(self.inst["required_tasks"]))

    def vocabulary(self):
        return "; ".join(sorted(self.label[c] for c in self.label))

    # ------------------------------------------------------------ decomposition
    def _decompose_prompt(self, feedback=""):
        return (f"GOAL: {self.inst['goal']['text']}\n"
                f"WHAT THE WORKFLOW MUST ACHIEVE: {self.brief()}\n"
                f"CAPABILITY VOCABULARY (use exactly one label per task): {self.vocabulary()}\n"
                f"{feedback}\n"
                "Decompose the goal into one task per required step. For each task choose the most general capability "
                "label that covers what the step requires (do not specialise beyond the step description). "
                "A task depends on the tasks that produce the data types it consumes. Return JSON: {\"tasks\": [{\"id\": \"t1\", \"capability\": "
                "\"<label from the vocabulary>\", \"description\": \"<what the task does>\", "
                "\"depends_on\": [\"<ids of tasks whose output this task consumes>\"]}]}")

    def _align(self, tasks):
        gold = self.inst["required_tasks"]
        pairs = []
        for i, t in enumerate(tasks):
            d = f"{t.get('description', '')} {t.get('capability', '')}"
            for g in gold:
                pairs.append((self.tfidf.cos(d, g["text"]), i, g["slot"]))
        pairs.sort(reverse=True)
        used_t, used_g, m = set(), set(), {}
        for s, i, g in pairs:
            if s < 0.15 or i in used_t or g in used_g:
                continue
            used_t.add(i)
            used_g.add(g)
            m[i] = g
        self.stats["extra_tasks"] += len(tasks) - len(m)
        return m

    def _to_plan(self, tasks):
        m = self._align(tasks)
        byslot = {t["slot"]: t for t in self.inst["required_tasks"]}
        idslot = {str(tasks[i].get("id", i)): s for i, s in m.items()}
        order = [t["slot"] for t in self.inst["required_tasks"]]
        planned = []
        for i, s in m.items():
            t = tasks[i]
            believed = self.ground(t.get("capability")) or byslot[s]["required_capability"]
            deps = [idslot[str(d)] for d in (t.get("depends_on") or []) if str(d) in idslot and idslot[str(d)] != s]
            gt = byslot[s]
            planned.append(dict(slot=s, believed=believed, text=gt["text"], task=gt, inputs=list(gt["inputs"]),
                                consumes=sorted(set(deps), key=order.index),
                                goal_input=gt["in_types"][0] if "goal" in gt["inputs"] else None))
        planned.sort(key=lambda p: order.index(p["slot"]))
        from . import oracle
        for p in planned:
            b, g = p["believed"], p["task"]["required_capability"]
            sub, sup = oracle.subsumed(self.dom, b, g), oracle.subsumed(self.dom, g, b)
            kind = ("label_exact" if b == g else "label_equivalent" if sub and sup else "label_specialised" if sub
                    else "label_generalised" if sup else "label_wrong")
            self.stats[kind] += 1
        self.stats["omitted"] += len(self.inst["required_tasks"]) - len(planned)
        return planned

    def decompose(self, feedback="", key="dec"):
        r = self._ask(SYSTEM_PLAN, self._decompose_prompt(feedback), 1500)
        tasks = (r or {}).get("tasks") or []
        if not isinstance(tasks, list):
            tasks = []
        if not tasks:
            self.stats["parse_fail_decompose"] += 1
        return self._to_plan([t for t in tasks if isinstance(t, dict)])

    def repair_decomposition(self, planned, missing_slots, key):
        byslot = {t["slot"]: t for t in self.inst["required_tasks"]}
        cur = [dict(id=p["slot"], capability=self.label.get(p["believed"], p["believed"]), description=p["text"],
                    depends_on=p["consumes"]) for p in planned]
        missing = ", ".join(byslot[s]["out_type"] for s in missing_slots if s in byslot)
        fb = (f"CURRENT PLAN: {json.dumps(cur)}\nVALIDATION VIOLATIONS: no task produces the required input(s): {missing}. "
              "Return the complete corrected plan.")
        r = self._ask(SYSTEM_PLAN, self._decompose_prompt(fb), 1500)
        tasks = [t for t in ((r or {}).get("tasks") or []) if isinstance(t, dict)]
        new = self._to_plan(tasks) if tasks else []
        have = {p["slot"]: p for p in planned}
        out = list(planned)
        for p in new:  # keep already planned tasks (and their assignments); add newly produced ones
            if p["slot"] not in have:
                out.append(p)
        order = [t["slot"] for t in self.inst["required_tasks"]]
        out.sort(key=lambda p: order.index(p["slot"]))
        kept = {p["slot"] for p in out}
        for p in out:
            if p["slot"] not in have:
                p["consumes"] = [x for x in p["consumes"] if x in kept]
        return out

    # ------------------------------------------------------------ assignment
    def choose(self, planned, candidates, upstream_types, info=("meta",), excluded=(), key="assign", attempt=0):
        if not candidates:
            return None, "no candidates"
        task = planned["task"]
        lines = []
        for a in candidates:
            ln = f"- {a['uri']}: {a['description']}"
            if "meta" in info:
                ln += (f" [capability={','.join(self.label.get(c, c) for c in a['capabilities'])}; input={','.join(a['accepts'])};"
                       f" output={a['produces']}; clearance={a['clearance']}; external={a['external']}; kind={a['kind']};"
                       f" cost={a['cost']}]")
            lines.append(ln)
        hist = ""
        if "history" in info:
            hist = "HISTORY (prior executions):\n" + "\n".join(
                f"{h['actor']}: {h['successes']}/{h['executions']} succeeded" for h in
                (self.hist.get(a["uri"]) for a in candidates) if h)
        meta = ""
        if "meta" in info:
            meta = (f"TASK REQUIREMENTS: required capability: {self.label.get(planned['believed'], planned['believed'])}; "
                    f"input types provided: {upstream_types}; required output type: {task['out_type']}; "
                    f"data sensitivity level: {task['sensitivity']}; personal data: {task['pii']}; "
                    f"budget per call: {task['max_cost']}; policy: {' | '.join(self.inst['policies']['rules'])}")
        user = PROMPTS["assign"].format(goal=self.inst["goal"]["text"], task=planned["text"], task_meta=meta,
                                        candidates="\n".join(lines), history=hist,
                                        feedback=("DO NOT CHOOSE (failed or rejected): " + ", ".join(sorted(excluded))) if excluded else "")
        uris = {a["uri"] for a in candidates}
        names = {a["uri"].split("#")[-1]: a["uri"] for a in candidates}
        for k in range(2):
            r = self._ask(SYSTEM_ASSIGN, user, 400)
            v = str((r or {}).get("actor", "")).strip().strip("<>")
            if v in uris:
                return v, str((r or {}).get("justification", ""))
            if v.split("#")[-1] in names:
                return names[v.split("#")[-1]], str((r or {}).get("justification", ""))
            self.stats["invalid_actor_answer"] += 1
            user += f"\nYour previous answer '{v[:80]}' is not one of the candidate URIs. Answer with a candidate URI."
        return None, "unparseable"
