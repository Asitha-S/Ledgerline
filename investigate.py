"""
investigate.py — explanation layer over the exception list.

Imports the existing modules unmodified. It reads audit records that controller.py has
already written and explains them. It does NOT match, rank, re-rank, propose, or revise
anything. The decision is a fact of the input; this only says why it happened.

    audit record (escalated row)
        -> structured evidence (nothing else)
        -> an LLM (Gemini or Claude — auto-detected from whichever key is set)
        -> explanation + recommended action + what would resolve it
        -> groundedness check against the evidence
        -> investigations.jsonl

HARD CONSTRAINTS (enforced, not merely requested)
-------------------------------------------------
1. The model never proposes, changes or ranks a match. The system prompt forbids it, the
   evidence contains no instruction to choose, and the output schema has no field that
   could carry a match. There is nowhere for a proposal to go.
2. Every number in the output must appear in the evidence. Amounts are supplied in BOTH
   decimal and integer-cent form precisely so the model never needs to convert units to
   cite a figure — which makes the groundedness test fair rather than a trap.
3. Ungrounded output is surfaced, not hidden. Each record carries its own
   `groundedness` block listing every numeric token that could not be traced to the
   evidence, and the run prints the count plainly at the end.

EVIDENCE PROVENANCE
-------------------
Everything comes from the audit record, except B's value date and reference fields,
which are looked up from the transactions CSV by B_id. That lookup is a join, not a
decision.

Added keys arrive with the probability that caused the completion classifier to accept
them, so the evidence can say not just WHICH key was added but how confident the
classifier was. Audit files written before that field existed carry bare key strings;
those are passed through as `probability: null, probability_available: false` rather
than back-filled with a guess.

PROVIDERS
---------
The prompt, the evidence block and the groundedness test are provider-neutral; only the
transport differs. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env and the provider is
chosen automatically, or force it with --provider.

Run:  python investigate.py [data_dir] [--n 50] [--provider gemini] [--model ...]
      python investigate.py --self-test      # offline checks, no API calls
      python investigate.py --list-models    # what the configured key can reach
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

import controller as CTL          # imported unmodified — for its class/trigger vocabulary

# Load .env if present, so the API key can live in a gitignored file rather than a shell
# export. Does not override a variable already set in the environment.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                override=False)
except ImportError:
    pass    # optional; the SDK still reads the environment directly

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


BATCHES = [
    ("BenchRec eval", "controller_audit_eval.jsonl", "BenchRec_cash_v1.0_eval.csv"),
    ("synthetic 50,000-group", "controller_audit_synth.jsonl", "synth_transactions.csv"),
]

SYSTEM_PROMPT = """\
You are writing reviewer-facing notes for a cash reconciliation exception queue.

A matching system has ALREADY decided that the transaction below cannot be closed
automatically. That decision is final and is not yours to revisit. Your only job is to
explain it to the human who has to work the exception.

ABSOLUTE CONSTRAINTS:

1. Do NOT propose a match. Do NOT say which candidate is correct, likely, best, or
   preferred. Do NOT rank, re-rank, score, or compare candidates in order to favour one.
   You may describe what the evidence shows about candidates; you may not conclude which
   one is right. If you catch yourself about to write "candidate 2 is probably the
   match", stop — that is the reviewer's call, not yours.
2. Do NOT recompute, derive, convert, or estimate any number. Every figure you write must
   appear VERBATIM in the evidence you were given. Amounts are provided in both decimal
   and cent form so you never need to convert. If a number you want is not in the
   evidence, describe the fact in words instead of inventing a figure.
3. Do NOT speculate about data you were not given. You do not know the ledger, the
   counterparty, or anything outside the evidence block.
4. If the evidence is insufficient to explain the escalation, say exactly that.

Write for a reconciliation analyst: direct, concrete, no hedging padding.
"""


class Investigation(BaseModel):
    """The three outputs. Note there is no field capable of carrying a proposed match."""
    explanation: str = Field(
        description="One paragraph explaining why this transaction could not be closed "
                    "automatically. Reference the specific triggers and evidence values."
    )
    recommended_action: str = Field(
        description="What the reviewer should DO next. A concrete action, not a guess at "
                    "the answer. Never name a candidate as the correct match."
    )
    information_needed: str = Field(
        description="The specific information that would resolve this exception — what "
                    "document, field, system, or confirmation is missing."
    )


# ----------------------------------------------------------------------------------
# Evidence assembly
# ----------------------------------------------------------------------------------
def _cents_to_amount(c):
    if c is None:
        return None
    return f"{c / 100:.2f}"


def build_evidence(rec: dict, b_row: dict | None) -> dict:
    """Exactly the fields specified, and nothing else. No labels, no gold answer, no
    hint about which candidate is right."""
    cands = []
    for c in rec.get("candidates", []):
        cands.append({
            "rank": c["rank"],
            "a_id": c["a_id"],
            "amount": _cents_to_amount(c["amount_cents"]),
            "amount_cents": c["amount_cents"],
            "delta_from_B": _cents_to_amount(c["amount_delta_cents"]),
            "delta_from_B_cents": c["amount_delta_cents"],
            "similarity_score": c["score"],
            "exact_amount_match": c["exact_amount"],
        })

    # controller.py now persists each added key with the probability that accepted it.
    # Older audit files carry bare strings; handle both rather than assume.
    added = []
    for k in rec.get("added_keys", []) or []:
        if isinstance(k, dict):
            added.append({"allocation_key": k.get("allocation_key"),
                          "probability": k.get("probability"),
                          "probability_available": k.get("probability") is not None})
        else:
            added.append({"allocation_key": k, "probability": None,
                          "probability_available": False})

    return {
        "transaction": {
            "b_id": rec["b_id"],
            "amount": _cents_to_amount(rec["b_amount_cents"]),
            "amount_cents": rec["b_amount_cents"],
            "value_date": (b_row or {}).get("B_valueDate"),
            "transaction_references": (b_row or {}).get("B_transactionReferences"),
            "transaction_attributes": (b_row or {}).get("B_transactionAttributes"),
        },
        "candidates_considered": cands,
        "candidate_pool_size_after_blocking": rec.get("pool_size"),
        "triggers_fired": rec.get("triggers", []),
        "exception_class": rec.get("exception_class"),
        "keys_added_by_completion_classifier": added,
        "duplicate_reference_among_candidates": rec.get("duplicate_reference_among_candidates"),
    }


# ----------------------------------------------------------------------------------
# Groundedness
# ----------------------------------------------------------------------------------
# A leading -/+ counts as a sign only when it is not glued to the preceding word.
# Without the lookbehind, "top-1" tokenises as "-1" and gets flagged as invented.
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


def _norm(tok: str) -> str:
    t = tok.replace(",", "").rstrip("%").lstrip("+")
    if t.endswith("."):
        t = t[:-1]
    try:
        f = float(t)
    except ValueError:
        return t
    return f"{f:.10g}"


def _collect_numbers(obj: Any, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out.add(_norm(str(obj)))
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            out.add(_norm(m))


def check_grounded(text: str, evidence: dict) -> dict:
    """Every numeric token in `text` must trace to the evidence.

    Allowances, each of which is a real derivation the model is entitled to make and
    none of which can launder an invented figure:
      * structural counts — candidate count, trigger count, added-key count, and ranks,
        all of which are determined by the evidence's own shape;
      * a rounded citation of an evidence number, matched at the precision written
        (0.12 for a score of 0.123456).
    """
    allowed = set()
    _collect_numbers(evidence, allowed)

    n_c = len(evidence.get("candidates_considered", []))
    for extra in (n_c, len(evidence.get("triggers_fired", [])),
                  len(evidence.get("keys_added_by_completion_classifier", []))):
        allowed.add(_norm(str(extra)))
    for r in range(0, n_c + 1):
        allowed.add(_norm(str(r)))

    raw_numeric = []
    for v in allowed:
        try:
            raw_numeric.append(float(v))
        except ValueError:
            pass

    ungrounded = []
    for tok in NUM_RE.findall(text or ""):
        n = _norm(tok)
        if n in allowed:
            continue
        # rounded citation of an evidence value, at the precision actually written
        try:
            f = float(n)
        except ValueError:
            ungrounded.append(tok)
            continue
        dec = len(n.split(".")[1]) if "." in n else 0
        if any(round(v, dec) == f for v in raw_numeric):
            continue
        ungrounded.append(tok)

    return {"grounded": not ungrounded, "ungrounded_tokens": ungrounded}


def check_no_match_proposed(text: str) -> dict:
    """Cheap tripwire for constraint 1. Flags for human review; does not edit output."""
    pats = [r"\bis (?:the|a) (?:correct|right|true) match\b",
            r"\bshould be matched to\b", r"\bmatch(?:es)? (?:to )?candidate\b",
            r"\bcandidate \d+ is (?:the )?(?:correct|right|best|likely)\b",
            r"\bI recommend matching\b", r"\bthe correct allocation is\b"]
    hits = [p for p in pats if re.search(p, text or "", re.I)]
    return {"clean": not hits, "matched_patterns": hits}


# ----------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------
def sample_ranked(data_dir, n_top, ranked_csv="exceptions_ranked_eval.csv",
                  audit="controller_audit_eval.jsonl",
                  tx="BenchRec_cash_v1.0_eval.csv", batch="BenchRec eval"):
    """Top-N escalated rows by exposure, in the order exposure.py ranked them.

    These are the rows a demo actually opens, so these are the ones that need
    explanations — unlike the stratified sample, which spread thin across classes.
    """
    q = pd.read_csv(os.path.join(data_dir, ranked_csv), dtype={"b_id": str})
    want = list(q.sort_values("rank")["b_id"].astype(str))[:n_top]
    order = {b: i for i, b in enumerate(want)}
    wanted = set(want)

    found = {}
    with open(os.path.join(data_dir, audit), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            b = str(d["b_id"])
            if b in wanted and d.get("decision") == "escalate":
                found[b] = d
    picked = [(batch, tx, found[b]) for b in want if b in found]

    b_lookup = {}
    path = os.path.join(data_dir, tx)
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         usecols=["B_id", "B_valueDate", "B_transactionReferences",
                                  "B_transactionAttributes"])
        df = df[df["B_id"].isin(wanted)]
        for r in df.to_dict("records"):
            b_lookup[(tx, r["B_id"])] = r
    return picked, b_lookup


def sample_escalated(data_dir, n_total, seed=0):
    """Stratified across every exception class actually present, across both batches."""
    by_class = {}
    b_lookup = {}
    for name, audit, tx in BATCHES:
        ap = os.path.join(data_dir, audit)
        if not os.path.exists(ap):
            continue
        with open(ap, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("decision") != "escalate":
                    continue
                by_class.setdefault(d["exception_class"], []).append((name, tx, d))

    classes = sorted(by_class)
    if not classes:
        return [], {}
    rng = random.Random(seed)
    per = max(1, n_total // len(classes))
    picked = []
    for c in classes:
        rows = by_class[c]
        rng.shuffle(rows)
        picked.extend(rows[:per])
    # top up to exactly n_total from the largest classes, round-robin
    i = 0
    while len(picked) < n_total:
        c = classes[i % len(classes)]
        pool = [r for r in by_class[c] if r not in picked]
        if pool:
            picked.append(pool[0])
        i += 1
        if i > len(classes) * 50:
            break
    picked = picked[:n_total]

    # B-side lookup only for the sampled ids
    need = {}
    for name, tx, d in picked:
        need.setdefault(tx, set()).add(str(d["b_id"]))
    for tx, ids in need.items():
        p = os.path.join(data_dir, tx)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, dtype=str, keep_default_na=False,
                         usecols=["B_id", "B_valueDate", "B_transactionReferences",
                                  "B_transactionAttributes"])
        df = df[df["B_id"].isin(ids)]
        for r in df.to_dict("records"):
            b_lookup[(tx, r["B_id"])] = r

    return picked, b_lookup


# ----------------------------------------------------------------------------------
# The call
# ----------------------------------------------------------------------------------
def _user_message(evidence: dict) -> str:
    return ("Explain this escalated reconciliation exception to the reviewer.\n\n"
            "EVIDENCE (this is everything you know; cite numbers only from here):\n"
            + json.dumps(evidence, indent=2))


# --- provider adapters -------------------------------------------------------------
# The prompt, the evidence and the groundedness test are provider-neutral. Only the
# transport differs, so each backend is a thin function returning an Investigation.

def _call_gemini(client, model, evidence, effort):
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=_user_message(evidence),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Investigation,
            temperature=0,          # explanation of fixed evidence; no reason to sample
            max_output_tokens=8000,  # thinking shares this budget; leave room
        ),
    )
    inv = getattr(resp, "parsed", None)
    if isinstance(inv, Investigation):
        return inv
    if inv is not None:                       # dict-shaped parse
        return Investigation.model_validate(inv)
    txt = (resp.text or "").strip()
    if not txt:
        raise RuntimeError("empty response (thinking may have consumed the output "
                           "budget); no parsed object and no text")
    return Investigation.model_validate_json(txt)         # last resort


# --- transient-failure handling ----------------------------------------------------
TRANSIENT_CODES = {429, 500, 502, 503, 504}


def _status_code(exc) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


# Connection-level failures carry no HTTP status but are just as transient as a 503.
# Without this, a single dropped socket looks permanent and kills the run.
_NETWORK_ERRORS = ("RemoteProtocolError", "ConnectError", "ConnectTimeout",
                   "ReadTimeout", "WriteTimeout", "PoolTimeout", "ReadError",
                   "WriteError", "ConnectionError", "ConnectionResetError",
                   "IncompleteRead", "ProtocolError", "TimeoutError",
                   "ServerDisconnectedError", "APIConnectionError", "SSLError")


def _is_network_error(exc) -> bool:
    names = {type(exc).__name__} | {c.__name__ for c in type(exc).__mro__}
    return bool(names & set(_NETWORK_ERRORS))


def _is_transient(exc) -> bool:
    return _status_code(exc) in TRANSIENT_CODES or _is_network_error(exc)


def call_with_retry(call, client, model, evidence, effort, attempts=4, base=2.0,
                    rate_limit_attempts=7, rate_limit_base=15.0, log=None):
    """Retry only transient failures. A 404 or 400 is a real error and is raised
    immediately — retrying it would waste time and money.

    429 (quota) is treated separately and far more patiently than a 5xx: on a
    ~10 req/min free tier the correct response to a quota refusal is to wait it out,
    not to give up on the row. Waits go 15s, 30s, 60s, 120s, 240s, 480s.
    """
    last = None
    a_429 = a_other = 0
    while True:
        try:
            return call(client, model, evidence, effort)
        except Exception as e:
            last = e
            code = _status_code(e)
            if not _is_transient(e):
                raise
            if code == 429:
                if a_429 >= rate_limit_attempts - 1:
                    raise
                wait = rate_limit_base * (2 ** a_429)
                a_429 += 1
                if log:
                    log(f"        429 quota — waiting {wait:.0f}s "
                        f"(attempt {a_429}/{rate_limit_attempts - 1})")
            else:
                if a_other >= attempts - 1:
                    raise
                wait = base * (2 ** a_other)
                a_other += 1
            time.sleep(wait)


def resolve_working_model(provider, client, ranked, effort, log=print):
    """Probe ranked models with one trivial call and take the first that answers.

    Necessary because the models endpoint lists ids that are not actually callable
    (observed: gemini-2.5-flash listed but returns 404) and because a brand-new model
    can be listed while its capacity is saturated (observed: gemini-3.7-flash, 503).
    """
    probe = {"transaction": {"b_id": "probe", "amount": "0.00", "amount_cents": 0},
             "candidates_considered": [], "triggers_fired": ["no_candidate"],
             "exception_class": "missing_counterparty",
             "keys_added_by_completion_classifier": []}
    for m in ranked:
        try:
            # Probing must fail FAST. The long 429 budget used for real rows would
            # stall here for ~15 minutes on a quota-refusing model before moving on,
            # when the right answer is simply to try the next model.
            call_with_retry(PROVIDERS[provider]["call"], client, m, probe, effort,
                            attempts=2, base=1.5,
                            rate_limit_attempts=1, rate_limit_base=2.0)
            return m, None
        except Exception as e:
            log(f"    {m}: unavailable ({type(e).__name__} "
                f"{_status_code(e)}) — trying next")
    return None, "no ranked model answered a probe call"


def _call_anthropic(client, model, evidence, effort):
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": _user_message(evidence)}],
        output_format=Investigation,
    )
    return resp.parsed_output


PROVIDERS = {
    "gemini": {
        "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "default_model": "gemini-2.5-pro",
        "call": _call_gemini,
    },
    "anthropic": {
        "env": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "default_model": "claude-opus-5",
        "call": _call_anthropic,
    },
}


def detect_provider():
    for name, spec in PROVIDERS.items():
        if any(os.environ.get(v) for v in spec["env"]):
            return name
    return None


def make_client(provider):
    if provider == "gemini":
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return genai.Client(api_key=key)
    import anthropic
    return anthropic.Anthropic()


def list_models(provider):
    client = make_client(provider)
    if provider == "gemini":
        out = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                out.append(m.name)
        return out
    return [m.id for m in client.models.list()]


# Model selection when no --model is given. A hardcoded id goes stale the moment the
# provider ships a new generation, so rank what the key actually reaches instead.
# Excluded: non-text modalities, and preview/experimental builds (tighter quotas).
_EXCLUDE = ("tts", "image", "embedding", "aqa", "gemma", "nano-banana", "omni",
            "vision", "live", "native-audio", "audio")
_PREVIEW = ("preview", "-exp", "experimental")


def _version_of(name: str):
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name.lower())
    return float(m.group(1)) if m else -1.0


def pick_model(provider, available):
    """Rank reachable models rather than hardcoding an id.

    Order: newest version first, then flash over pro (this workload is 50 short
    structured-output calls — flash is the right tool and the cheaper one), then
    full over lite. Preview builds are used only if nothing stable is reachable.
    """
    if not available:
        return None, "no reachable models returned"

    usable = [m for m in available if not any(x in m.lower() for x in _EXCLUDE)]
    if not usable:
        return available[0], "no text model identified; using first reachable"

    stable = [m for m in usable if not any(x in m.lower() for x in _PREVIEW)]
    pool, note = (stable, "") if stable else (usable, " (only preview builds reachable)")

    def rank(name):
        n = name.lower()
        return (_version_of(n),
                1 if "flash" in n else 0,      # prefer flash
                0 if "lite" in n else 1,       # prefer full over lite
                -len(n))                       # prefer the plainer name
    ordered = sorted(pool, key=rank, reverse=True)
    return ordered, f"ranked {len(ordered)} stable text models{note}"


def _fmt(inv: Investigation, evidence: dict, gr: dict, mp: dict, batch: str) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"B_id {evidence['transaction']['b_id']}   [{batch}]   "
                 f"exception_class = {evidence['exception_class']}")
    lines.append("=" * 100)
    lines.append("")
    lines.append("--- EVIDENCE PASSED TO THE MODEL ---")
    lines.append(json.dumps(evidence, indent=2))
    lines.append("")
    lines.append("--- EXPLANATION ---")
    lines.append(inv.explanation)
    lines.append("")
    lines.append("--- RECOMMENDED ACTION ---")
    lines.append(inv.recommended_action)
    lines.append("")
    lines.append("--- INFORMATION NEEDED ---")
    lines.append(inv.information_needed)
    lines.append("")
    lines.append(f"--- CHECKS ---   grounded={gr['grounded']}   "
                 f"no_match_proposed={mp['clean']}")
    if gr["ungrounded_tokens"]:
        lines.append(f"    ungrounded numeric tokens: {gr['ungrounded_tokens']}")
    if mp["matched_patterns"]:
        lines.append(f"    match-proposal patterns hit: {mp['matched_patterns']}")
    return "\n".join(lines)


def _self_test():
    """Validates evidence assembly and both checkers without touching the API."""
    print("SELF-TEST — evidence assembly and checkers (no API calls)")
    print("-" * 100)
    ev = {"transaction": {"amount": "-20893751.85", "amount_cents": -2089375185},
          "candidates_considered": [
              {"rank": 1, "similarity_score": 0.123456, "amount": "-20893751.85"},
              {"rank": 2, "similarity_score": 0.4, "amount": "-100.00"}],
          "triggers_fired": ["completion_added"],
          "keys_added_by_completion_classifier": []}

    cases = [
        ("cites an exact evidence figure", "The amount -20893751.85 matched.", True),
        ("cites a rounded score", "Top-1 scored 0.12 here.", True),
        ("cites a structural count", "There were 2 candidates.", True),
        ("invents a figure", "The fee was 3.5% of 88888.00.", False),
        ("invents a percentage", "That is 47% of the total.", False),
        ("no numbers at all", "The amounts differ materially.", True),
    ]
    ok = True
    for name, text, expect in cases:
        got = check_grounded(text, ev)["grounded"]
        flag = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{flag}] {name:34s} grounded={got} (expected {expect})")

    mcases = [("neutral description", "Two candidates share a reference string.", True),
              ("proposes a match", "Candidate 2 is the correct match.", False),
              ("recommends matching", "I recommend matching to the second row.", False)]
    for name, text, expect in mcases:
        got = check_no_match_proposed(text)["clean"]
        flag = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{flag}] {name:34s} clean={got} (expected {expect})")
    print("-" * 100)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?",
                    default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--provider", choices=sorted(PROVIDERS),
                    help="default: whichever key is present in the environment/.env")
    ap.add_argument("--model", default=None,
                    help="default: the provider's default model")
    ap.add_argument("--effort", default="medium", help="anthropic only; ignored by gemini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list-models", action="store_true",
                    help="print model ids the configured key can reach, then exit")
    ap.add_argument("--rank-top", type=int, default=None, metavar="N",
                    help="explain the top N eval exceptions by exposure, in the order "
                         "exceptions_ranked_eval.csv gives, instead of a stratified sample")
    ap.add_argument("--rpm", type=float, default=10.0,
                    help="target requests per minute (free tier is ~10)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace investigations.jsonl instead of appending; by default "
                         "existing rows are kept and b_ids already present are skipped")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if _self_test() else 1)

    provider = args.provider or detect_provider()
    if provider is None:
        print("No API key found for any supported provider.")
        print("  Gemini    : set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        print("  Anthropic : set ANTHROPIC_API_KEY")
        print("Put it in a .env file next to this script (cp .env.example .env).")
        print(".env is gitignored. Aborting before spending anything.")
        return
    model = args.model
    auto_note = ""
    _probe_client = None
    if model is None:
        try:
            avail = list_models(provider)
            ranked, why = pick_model(provider, avail)
            if not ranked:
                print(f"No usable model found for {provider}. Run --list-models.")
                return
            print(f"  probing {len(ranked)} ranked models for one that answers "
                  f"({why})...")
            _probe_client = make_client(provider)
            model, err = resolve_working_model(provider, _probe_client, ranked[:6],
                                               args.effort, log=print)
            if model is None:
                print(f"  {err}. Run --list-models, then pass --model explicitly.")
                return
            auto_note = f"  (auto-selected from {len(avail)} reachable models)"
        except Exception as e:
            print(f"  Model resolution failed: {type(e).__name__}: {e}")
            return

    if args.list_models:
        try:
            for m in list_models(provider):
                print(m)
        except Exception as e:
            print(f"Could not list models: {type(e).__name__}: {e}")
        return

    print("=" * 100)
    print("INVESTIGATE — explanation layer over the exception list")
    print("=" * 100)
    print()
    print("  This layer explains decisions. It does not make or change them.")
    print("  Evidence passed to the model contains no labels and no gold answer.")
    print()

    if args.rank_top:
        picked, b_lookup = sample_ranked(args.data_dir, args.rank_top)
        selection = f"top {args.rank_top} eval exceptions by exposure (ranked order)"
    else:
        picked, b_lookup = sample_escalated(args.data_dir, args.n, args.seed)
        selection = f"stratified sample of {args.n} across exception classes"
    if not picked:
        print("  No escalated audit records found. Run controller.py first.")
        return

    # Append-with-dedup: keep every existing record, skip b_ids already explained.
    out_path = os.path.join(args.data_dir, "investigations.jsonl")
    existing, existing_ids = [], set()
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:          # a prior failure is not a result; allow retry
                    continue
                if str(d.get("b_id")) in existing_ids:
                    continue
                existing.append(d)
                existing_ids.add(str(d["b_id"]))
    before = len(picked)
    picked = [p for p in picked if str(p[2]["b_id"]) not in existing_ids]
    skipped = before - len(picked)

    dist = {}
    for _, _, d in picked:
        dist[d["exception_class"]] = dist.get(d["exception_class"], 0) + 1
    print(f"  selection: {selection}")
    print(f"  existing records kept: {len(existing)}   "
          f"already explained, skipped: {skipped}")
    print(f"  to explain now: {len(picked)}  by class: {dist}")
    if not picked:
        print("  Nothing new to do.")
        return
    print(f"  provider={provider}  model={model}"
          + (f"  effort={args.effort}" if provider == "anthropic" else ""))
    if auto_note:
        print(f"  {auto_note.strip()}")
        print("  override with --model <id>; see --list-models for the full list")
    print()

    try:
        client = _probe_client or make_client(provider)
    except Exception as e:
        print(f"  Could not construct {provider} client: {type(e).__name__}: {e}")
        return
    call = PROVIDERS[provider]["call"]

    results, n_ungrounded, n_proposed, n_failed = [], 0, 0, 0
    fail_reasons = {}
    models_used = {}
    consecutive_quota_failures = 0
    QUOTA_CIRCUIT_BREAK = 2   # sustained 429 after full backoff == quota wall, not a spike
    min_gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    print(f"  pacing: {args.rpm:g} req/min -> {min_gap:.1f}s between calls; "
          f"429 backs off up to ~8min rather than dropping a row")
    print()
    last_call = 0.0

    with open(out_path, "w", encoding="utf-8") as fh:
        for d in existing:                      # preserve every prior record verbatim
            fh.write(json.dumps(d) + "\n")
        for i, (batch, tx, rec) in enumerate(picked, start=1):
            gap = min_gap - (time.monotonic() - last_call)
            if gap > 0 and i > 1:
                time.sleep(gap)
            last_call = time.monotonic()
            ev = build_evidence(rec, b_lookup.get((tx, str(rec["b_id"]))))
            try:
                inv = call_with_retry(call, client, model, ev, args.effort,
                                      log=lambda m: print(m, flush=True))
            except Exception as e:
                n_failed += 1
                code = _status_code(e)
                reason = f"{type(e).__name__} {code}" if code else type(e).__name__
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                fh.write(json.dumps({"b_id": rec["b_id"], "batch": batch,
                                     "evidence": ev, "error": f"{type(e).__name__}: {e}"}) + "\n")
                fh.flush()
                print(f"  [{i}/{len(picked)}] b_id {rec['b_id']}: FAILED — {reason}")
                if code == 429:
                    consecutive_quota_failures += 1
                    if consecutive_quota_failures >= QUOTA_CIRCUIT_BREAK:
                        print()
                        print(f"  QUOTA WALL: {consecutive_quota_failures} rows in a row "
                              f"exhausted the full 429 backoff (~16 min each).")
                        print("  That is a sustained quota limit, not a transient spike — "
                              "waiting longer")
                        print("  will not help today. Stopping with "
                              f"{len(results)} of {len(picked)} done.")
                        print("  Rerun the same command when quota resets: completed b_ids")
                        print("  are skipped automatically, so it resumes where it stopped.")
                        break
                else:
                    consecutive_quota_failures = 0
                # Stop early only if the failure looks like misconfiguration (a bad
                # model id, bad auth). A transient network blip must not kill the run.
                if i == 1 and code is not None and 400 <= code < 500 and code != 429:
                    print()
                    print(f"  First call failed with {code} — that is a configuration")
                    print("  error, not a transient one. Stopping. Try --model <id>.")
                    break
                continue
            models_used[model] = models_used.get(model, 0) + 1
            consecutive_quota_failures = 0

            joined = " ".join([inv.explanation, inv.recommended_action,
                               inv.information_needed])
            gr = check_grounded(joined, ev)
            mp = check_no_match_proposed(joined)
            if not gr["grounded"]:
                n_ungrounded += 1
            if not mp["clean"]:
                n_proposed += 1

            rowout = {
                "b_id": rec["b_id"], "batch": batch,
                "exception_class": ev["exception_class"],
                "triggers_fired": ev["triggers_fired"],
                "evidence": ev,
                "explanation": inv.explanation,
                "recommended_action": inv.recommended_action,
                "information_needed": inv.information_needed,
                "groundedness": gr,
                "no_match_proposed_check": mp,
                "provider": provider, "model": model, "effort": args.effort,
            }
            fh.write(json.dumps(rowout) + "\n")
            fh.flush()      # a long paced run must survive interruption; without this,
                            # rows already paid for sit in the buffer and are lost
            results.append((batch, ev, inv, gr, mp))
            print(f"  [{i}/{len(picked)}] b_id {rec['b_id']}  {ev['exception_class']:30s} "
                  f"grounded={gr['grounded']}")

    print()
    print("=" * 100)
    print("FIVE EXAMPLES IN FULL")
    print("=" * 100)
    seen, shown = set(), 0
    for batch, ev, inv, gr, mp in results:          # one per class first, then fill
        if shown >= 5:
            break
        if ev["exception_class"] in seen:
            continue
        seen.add(ev["exception_class"])
        print()
        print(_fmt(inv, ev, gr, mp, batch))
        shown += 1
    for batch, ev, inv, gr, mp in results:
        if shown >= 5:
            break
        print()
        print(_fmt(inv, ev, gr, mp, batch))
        shown += 1

    print()
    print("=" * 100)
    print("GROUNDEDNESS")
    print("=" * 100)
    print()
    n = len(results)
    print("  THIS RUN (new rows only)")
    pct = f"   ({n_ungrounded / n * 100:.2f}%)" if n else ""
    print(f"    attempted                                            {len(picked):>6,}")
    print(f"    succeeded                                            {n:>6,}")
    print(f"    failed                                               {n_failed:>6,}")
    if fail_reasons:
        for r, c in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"        {r:<44} {c:>6,}")
    print(f"    containing a number NOT in the evidence              {n_ungrounded:>6,}{pct}")
    print(f"    tripping the no-match-proposed tripwire              {n_proposed:>6,}")
    if models_used:
        print("    model(s) that actually answered:")
        for m, c in sorted(models_used.items(), key=lambda x: -x[1]):
            print(f"        {m:<44} {c:>6,} rows")
    if n == 0:
        print()
        print("    No explanations produced, so there is nothing to measure here. The")
        print("    count above is 0 because the set is empty, NOT because it was clean.")
    if n_ungrounded:
        print()
        print("    Ungrounded instances (surfaced, not suppressed):")
        for batch, ev, inv, gr, mp in results:
            if not gr["grounded"]:
                print(f"      b_id {ev['transaction']['b_id']}  {gr['ungrounded_tokens']}")

    # Combined: re-read the file so the figure covers every record now on disk,
    # including the earlier run's, which may have used a different model.
    print()
    print("  COMBINED (every record in investigations.jsonl)")
    combined, by_model = [], {}
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            combined.append(d)
            by_model[d.get("model") or "?"] = by_model.get(d.get("model") or "?", 0) + 1
    c_un = [d for d in combined if not d.get("groundedness", {}).get("grounded", True)]
    c_mp = [d for d in combined
            if not d.get("no_match_proposed_check", {}).get("clean", True)]
    cn = len(combined)
    print(f"    explanations on disk                                 {cn:>6,}")
    print(f"    containing a number NOT in the evidence              {len(c_un):>6,}"
          f"   ({len(c_un) / cn * 100:.2f}%)" if cn else "")
    print(f"    tripping the no-match-proposed tripwire              {len(c_mp):>6,}")
    print("    by model:")
    for m, c in sorted(by_model.items(), key=lambda x: -x[1]):
        n_u = sum(1 for d in combined
                  if (d.get("model") or "?") == m
                  and not d.get("groundedness", {}).get("grounded", True))
        print(f"        {m:<44} {c:>6,} rows, {n_u} ungrounded")
    if c_un:
        print()
        print("    All ungrounded instances on disk:")
        for d in c_un:
            print(f"      b_id {d['b_id']}  {d['groundedness']['ungrounded_tokens']}"
                  f"   [{d.get('model')}]")
    print()
    n_with_p = sum(1 for _, ev, *_ in results
                   for a in ev["keys_added_by_completion_classifier"]
                   if a["probability_available"])
    n_without = sum(1 for _, ev, *_ in results
                    for a in ev["keys_added_by_completion_classifier"]
                    if not a["probability_available"])
    print(f"  added-key probabilities present: {n_with_p}   missing: {n_without}")
    if n_without:
        print("  (missing ones come from audit files written before controller.py")
        print("   persisted the field; rerun controller.py to populate them)")
    print()
    print(f"  results -> {out_path}")


if __name__ == "__main__":
    _main()
