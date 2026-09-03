"""
export.py — convert existing outputs into static JSON for a front end.

This is a serialiser. It makes no decisions, calls no model, and contains no analysis
logic beyond arithmetic over figures already produced: counts, sums, percentages and
the cumulative curve. Every decision, answer, trigger and probability is copied
verbatim from the controller's audit records.

INPUTS
    controller_audit_{eval,synth}.jsonl   decisions, candidates, triggers, added keys
    exceptions_ranked_{eval,synth}.csv    the exposure-ranked queue
    investigations.jsonl                  LLM explanations + grounding results
    {BenchRec_cash_v1.0,synth}_solution.csv   labels, for correctness only
    ctrl.log                              throughput only (a runtime measurement that
                                          cannot be derived from an audit record)

OUTPUTS  (web/data/)
    summary_{eval,synth}.json
    queue_{eval,synth}.json
    curve_{eval,synth}.json
    detail/{eval,synth}/<b_id>.json       one per escalated row

CONSISTENCY GUARANTEE
    Summary and curve figures are computed here from the audit files directly —
    exposure.log is never parsed. They are then cross-checked against exposure.py's own
    implementation, invoked in-process. Two independent code paths reading the same
    inputs must agree; any disagreement aborts the export rather than shipping a number
    that contradicts the analysis.

Run:  python export.py [data_dir]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

import controller as CTL          # trigger / exception-class vocabulary
import exposure as EXP            # cross-check reference implementation
from score import _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BATCHES = [
    {"key": "eval", "name": "BenchRec eval",
     "audit": "controller_audit_eval.jsonl",
     "ranked": "exceptions_ranked_eval.csv",
     "solution": "BenchRec_cash_v1.0_solution.csv",
     "transactions": "BenchRec_cash_v1.0_eval.csv",
     "currency": "USD"},
    {"key": "synth", "name": "synthetic 50,000-group",
     "audit": "controller_audit_synth.jsonl",
     "ranked": "exceptions_ranked_synth.csv",
     "solution": "synth_solution.csv",
     "transactions": "synth_transactions.csv",
     "currency": "USD"},
]

OUT_ROOT = os.path.join("web", "data")

# The queue is written in pages. A single synthetic queue file is 2.1 MB, which the
# browser has to fetch whole before it can draw a row; paged, the first fetch is a
# tenth of that and the rest arrives behind the already-usable table. Rows are ranked
# by exposure, so page 0 is the part of the queue anyone looks at first.
QUEUE_PAGE = 1000
LARGE_FILE_BYTES = 1_000_000      # flag anything a browser would be slow to fetch
TOL = 1e-6


def _log(m=""):
    print(m, flush=True)


def _rule(c="-", n=100):
    _log(c * n)


# ----------------------------------------------------------------------------------
# Load — independent of exposure.py on purpose, so the cross-check means something
# ----------------------------------------------------------------------------------
def load_audit(data_dir, batch):
    sol = pd.read_csv(os.path.join(data_dir, batch["solution"]), dtype=str,
                      keep_default_na=False)
    labels = {str(b): _parse_alloc(t)
              for b, t in zip(sol["B_id"], sol["targetAllocation"])}

    recs = []
    with open(os.path.join(data_dir, batch["audit"]), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            bid = str(d["b_id"])
            answer = set(d.get("answer_keys") or [])
            d["_b_id"] = bid
            d["_exposure_cents"] = abs(int(d["b_amount_cents"]))
            d["_auto"] = d["decision"] == "auto_close"
            d["_correct"] = answer == labels.get(bid, set())
            d["_gold_blank"] = len(labels.get(bid, set())) == 0
            recs.append(d)
    return recs


def load_b_side(data_dir, batch):
    """Value/import dates and reference fields for B rows. A join, not a computation —
    the audit record carries the amount but not the dates."""
    p = os.path.join(data_dir, batch.get("transactions", ""))
    if not p or not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype=str, keep_default_na=False,
                     usecols=["B_transactionType", "B_id", "B_valueDate", "B_importDate",
                              "B_currencyCode", "B_debitOrCredit",
                              "B_transactionReferences", "B_transactionAttributes"])
    df = df[df["B_transactionType"] == "B"]
    return {r["B_id"]: r for r in df.to_dict("records")}


def load_investigations(data_dir):
    p = os.path.join(data_dir, "investigations.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if "error" in d:
                continue
            out[str(d["b_id"])] = {
                "explanation": d.get("explanation"),
                "recommended_action": d.get("recommended_action"),
                "information_needed": d.get("information_needed"),
                "grounded": d.get("groundedness", {}).get("grounded"),
                "ungrounded_tokens": d.get("groundedness", {}).get("ungrounded_tokens", []),
                "no_match_proposed": d.get("no_match_proposed_check", {}).get("clean"),
                "provider": d.get("provider"),
                "model": d.get("model"),
            }
    return out


def throughput_from_ctrl_log(data_dir):
    """Throughput is a runtime measurement; it cannot be recovered from an audit record.
    Read from the controller's own run log, or null if unavailable. (exposure.log is
    never read — that prohibition is about the analysis figures, which are recomputed.)"""
    p = os.path.join(data_dir, "ctrl.log")
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for name, key in (("BenchRec eval", "eval"), ("synthetic 50,000-group", "synth")):
        # The heading is followed by its own rule line, so skip that before capturing;
        # anchoring on the first "====" after the title matches an empty block.
        m = re.search(r"FINAL REPORT — " + re.escape(name) + r"\s*\n=+\n(.*?)(?=\n={10,}|\Z)",
                      txt, re.S)
        if not m:
            continue
        blk = m.group(1)
        t = re.search(r"throughput\s+([\d,\.]+)\s+records/sec", blk)
        w = re.search(r"wall clock\s+([\d\.]+)\s*s", blk)
        out[key] = {
            "records_per_sec": float(t.group(1).replace(",", "")) if t else None,
            "wall_clock_sec": float(w.group(1)) if w else None,
        }
    return out


def reference_from_score_log(data_dir):
    """The third-party prediction file's score, read from score.py's own output.

    Not recomputed here and not typed in: score.log is the record of scoring
    MatcherByChatGPT_submission.csv against the eval solution, and the interface
    shows exactly what that run reported. Returns None if the log is absent, in
    which case the comparison is simply not shown."""
    p = os.path.join(data_dir, "score.log")
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    name = re.search(r"SCORE\s+—\s+(\S+)\s+vs\s+(\S+)", txt)
    blk = re.search(r"\nOVERALL\n(.*?)(?=\n[A-Z][A-Z ]{3,}\n|\Z)", txt, re.S)
    if not blk:
        return None
    b = blk.group(1)

    def num(label, cast=float):
        m = re.search(re.escape(label) + r"\s+([\d,\.]+)", b)
        return cast(m.group(1).replace(",", "")) if m else None

    out = {
        "source_file": name.group(1) if name else "MatcherByChatGPT_submission.csv",
        "scored_against": name.group(2) if name else None,
        "total_rows": num("total rows", int),
        "predicted": num("predicted", int),
        "correct": num("correct", int),
        "match_rate_pct": num("match rate"),
        "match_precision_pct": num("match precision"),
        "abstention_rate_pct": num("abstention rate"),
        "provenance": ("A prediction file found alongside the dataset. Not an official "
                       "published baseline, and it carries no provenance — shown because "
                       "it is the only other answer set in the folder."),
        "from": "score.log",
    }
    return out if out["match_rate_pct"] is not None else None


GATE_LABELS = {
    "exact keys > 1": ("multi_exact", "More than one exact-amount candidate key"),
    "dup_ref":        ("dup_ref",     "Duplicate reference among candidates"),
    "keys>1 or dup":  ("either",      "Either rule fires"),
}


def gates_from_driftgate_log(data_dir, batch_name):
    """The gate pricing table driftgate.py printed, for one batch.

    driftgate.py measures what each candidate control would have cost had it been
    applied to decisions already made. Nothing here re-decides anything: the numbers
    are lifted verbatim so the interface can show the trade instead of describing it.

    Only three of the five gates driftgate.py prices are carried, because they are the
    three the interface offers: the multi-exact-key rule, the duplicate-reference rule,
    and their union. The row-count variant of the first is deliberately left out — it
    counts candidate ROWS rather than allocation keys, and driftgate.py explains at
    length why that is the wrong count. Returns None if the log is absent."""
    fp = os.path.join(data_dir, "driftgate.log")
    if not os.path.exists(fp):
        return None
    txt = open(fp, encoding="utf-8", errors="replace").read()

    # each batch has its own PRICING section; take the one under this batch's heading
    i = txt.find("POOL STRUCTURE vs DRIFT OUTCOME — " + batch_name)
    if i < 0:
        return None
    j = txt.find("POOL STRUCTURE vs DRIFT OUTCOME", i + 1)
    blk = txt[i:j if j > 0 else len(txt)]

    base = re.search(r"current\s+coverage ([\d.]+)%\s+precision ([\d.]+)%\s+"
                     r"\(([\d,]+) of ([\d,]+) auto-closed, ([\d,]+) correct\)", blk)
    if not base:
        return None

    n = lambda x: int(x.replace(",", ""))
    states = [{
        "key": "none", "label": "No gate (shipped)", "shipped": True,
        "escalates": 0, "wrong_caught": 0, "zero_drift_caught": 0,
        "nonzero_drift_caught": 0, "bounded_caught": 0, "correct_given_up": 0,
        "correct_per_wrong": None,
        "coverage_pct": float(base.group(1)), "precision_pct": float(base.group(2)),
        "auto_closed": n(base.group(3)), "records": n(base.group(4)),
        "correct": n(base.group(5)),
    }]

    #  gate            escalates wrong d=0 d!=0 bnd  correct_esc ok/wrong coverage precision pts/pt
    row = re.compile(r"^  (.{1,16}?)\s{2,}([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+"
                     r"([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                     r"([\d.]+|n/a)\s*$", re.M)
    found = {}
    for m in row.finditer(blk):
        name = m.group(1).strip()
        if name not in GATE_LABELS:
            continue
        key, label = GATE_LABELS[name]
        found[key] = {
            "key": key, "label": label, "rule": name, "shipped": False,
            "escalates": n(m.group(2)), "wrong_caught": n(m.group(3)),
            "zero_drift_caught": n(m.group(4)), "nonzero_drift_caught": n(m.group(5)),
            "bounded_caught": n(m.group(6)), "correct_given_up": n(m.group(7)),
            "correct_per_wrong": float(m.group(8)),
            "coverage_pct": float(m.group(9)), "precision_pct": float(m.group(10)),
            "points_per_point": None if m.group(11) == "n/a" else float(m.group(11)),
        }
    if len(found) != 3:
        return None
    for k in ("multi_exact", "dup_ref", "either"):
        states.append(found[k])

    # how many invisible errors exist at all, so the catch can be shown as a fraction
    z = re.search(r"zero-drift wrong postings \(invisible to a balance check\)\s+([\d,]+)",
                  blk)
    return {
        "batch": batch_name,
        "invisible_total": n(z.group(1)) if z else None,
        "states": states,
        "note": ("Tested against decisions already made and NOT adopted. The toggle is "
                 "here so the trade can be seen rather than described."),
    }


def operating_points_from_complete_log(data_dir):
    """complete.py's completion-threshold sweep, 0.50 to 0.95, plus completion OFF.

    The shipped operating point is 0.50. complete.py prints the table and explicitly
    declines to pick a row; this lifts it so the interface can let a reader move along
    it. Measured on BenchRec eval only — there is no comparable sweep for a batch we
    generated, so it rides with eval. Returns None if the table is absent."""
    fp = os.path.join(data_dir, "complete.log")
    if not os.path.exists(fp):
        return None
    txt = open(fp, encoding="utf-8", errors="replace").read()
    i = txt.find("threshold  overall_match_%")
    if i < 0:
        return None

    pts, off = [], None
    for line in txt[i:].split("\n")[1:]:
        m = re.match(r"\s*(OFF|[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s*$", line)
        if not m:
            break
        rec = {
            "overall_match_pct": float(m.group(2)),
            "overall_precision_pct": float(m.group(3)),
            "single_match_pct": float(m.group(4)),
            "single_precision_pct": float(m.group(5)),
            "multi_match_pct": float(m.group(6)),
            "gained": int(m.group(7)), "lost": int(m.group(8)), "net": int(m.group(9)),
        }
        if m.group(1) == "OFF":
            rec["threshold"] = None
            off = rec
        else:
            rec["threshold"] = float(m.group(1))
            rec["shipped"] = rec["threshold"] == 0.5
            pts.append(rec)
    if not pts or off is None:
        return None

    # The flat band, derived rather than asserted. TOL is deliberately tight: across
    # 0.50-0.65 the match rate moves 0.0156 pts and the next step out is 0.0562, a
    # 3.6x jump. 0.02 separates those cleanly without being fitted to an answer.
    TOL = 0.02
    top = pts[0]["overall_match_pct"]
    band = [p["threshold"] for p in pts if abs(p["overall_match_pct"] - top) <= TOL]
    band_end = band[0]
    for t in band:
        if all(abs(q["overall_match_pct"] - top) <= TOL
               for q in pts if pts[0]["threshold"] <= q["threshold"] <= t):
            band_end = t
    inband = [p for p in pts if p["threshold"] <= band_end]

    return {
        "measured_on": "BenchRec eval",
        "shipped_threshold": 0.5,
        "completion_off": off,
        "points": pts,
        "flat_band": {
            "from": pts[0]["threshold"], "to": band_end,
            "spread_pts": round(max(p["overall_match_pct"] for p in inband)
                                - min(p["overall_match_pct"] for p in inband), 4),
            "note": ("Overall match rate varies by 0.0156 points across this band, so "
                     "the operating point is a range rather than an optimum. "
                     "complete.py prints the table and declines to pick a row."),
        },
    }


def regimes_from_findings_log(data_dir):
    """The multi-key regime split on both splits, read out of findings.py's output.

    Nothing here is recomputed: findings.py measures the regimes, validates the eval
    reconstruction against train's matchId groups, and prints the worked example. This
    lifts those numbers verbatim so the interface can show them. Returns None if the
    log is absent or does not contain both splits, in which case the section is simply
    not rendered."""
    p = os.path.join(data_dir, "findings.log")
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()

    def split_block(after):
        """The 'regime, per B row (%)' table that follows a heading."""
        i = txt.find(after)
        if i < 0:
            return None
        j = txt.find("regime, per B row (%):", i)
        if j < 0:
            return None
        out = {}
        for line in txt[j:].split("\n")[2:]:
            m = re.match(r"\s*(repeat|neither|partition)\s+([\d.]+)\s*$", line)
            if not m:
                break
            out[m.group(1)] = float(m.group(2))
        return out or None

    train = split_block("multi-key groups repeat rather than partition (train)")
    ev = split_block("multi-key regime on eval (groups reconstructed from the labels)")
    if not train or not ev:
        return None

    def num(pattern, cast=float):
        m = re.search(pattern, txt)
        return cast(m.group(1).replace(",", "")) if m else None

    # the worked example, parsed out of the fixed-width tables findings.py prints
    ex = None
    m = re.search(r"WORKED EXAMPLE — eval B_id (\d+) \(repeat regime\)([\s\S]*?)"
                  r"target set\s+\[([^\]]*)\]", txt)
    if m:
        body = m.group(2)

        def rows(header, cols):
            k = body.find(header)
            if k < 0:
                return []
            out = []
            for line in body[k:].split("\n")[2:]:
                f = line.split()
                if len(f) < cols:
                    break
                out.append(f)
            return out

        b_rows = [{"b_id": f[0], "dc": f[1], "amount": f[2]}
                  for f in rows("B rows (external statement):", 3)]
        a_rows = [{"a_id": f[0], "dc": f[1], "amount": f[2], "key": f[3]}
                  for f in rows("A rows (internal ledger):", 4)]
        eq = re.search(r"A rows with amount == B\s+(\d+) of (\d+)", body)
        ex = {
            "b_id": m.group(1),
            "statement_rows": b_rows,
            "ledger_rows": a_rows,
            "b_amount": (re.search(r"B amount\s+([\d,\.]+)", body) or [None, None])[1],
            "sum_of_ledger_amounts":
                (re.search(r"sum of all A amounts\s+([\d,\.]+)", body) or [None, None])[1],
            "ledger_rows_at_b_amount": int(eq.group(1)) if eq else None,
            "ledger_rows_total": int(eq.group(2)) if eq else None,
            "keys": [k.strip().strip("\'") for k in m.group(3).split(",")],
        }

    return {
        "from": "findings.log",
        "train": train,
        "eval": ev,
        "multi_key_rows": {
            "train": num(r"multi-key B rows in train: ([\d,]+)", int),
            "eval": num(r"multi-key B rows in eval: ([\d,]+)", int),
        },
        "eval_reconstruction": {
            "regime_agreement_pct": num(r"regime unchanged: [\d,]+ of [\d,]+ \(([\d.]+)%\)"),
            "over_collected_pct": num(r"over-collected  \(picked up A rows from other "
                                      r"groups\): [\d,]+ \(([\d.]+)%\)"),
            "note": ("eval has no matchId, so a group's ledger rows are reconstructed from "
                     "the keys its label names. Validated against train's matchId groups: "
                     "the reconstruction never loses a row, over-collects on some, and "
                     "leaves the regime label unchanged on the share above."),
        },
        "worked_example": ex,
    }


# ----------------------------------------------------------------------------------
# Figures — arithmetic over already-decided rows
# ----------------------------------------------------------------------------------
def compute_summary(recs, name, throughput, currency="USD"):
    n = len(recs)
    auto = [r for r in recs if r["_auto"]]
    esc = [r for r in recs if not r["_auto"]]
    ok = [r for r in auto if r["_correct"]]
    bad = [r for r in auto if not r["_correct"]]

    # What the batch would have scored with no controller at all: every proposed
    # answer posted blind. This is the figure the reference file is comparable to,
    # since that file also has no notion of escalating.
    #
    # `predicted` follows score.py's rule exactly: a row counts as abstained only when
    # the prediction is empty AND the gold label is not. Proposing nothing against a
    # blank label is an answer, and a correct one — counting it as an abstention put
    # this 0.5 points out and the consistency gate caught it against ctrl.log.
    correct_all = [r for r in recs if r["_correct"]]
    abstained_all = [r for r in recs
                     if not r.get("answer_keys") and not r.get("_gold_blank")]
    n_predicted = n - len(abstained_all)

    tot_v = sum(r["_exposure_cents"] for r in recs)
    auto_v = sum(r["_exposure_cents"] for r in auto)
    ok_v = sum(r["_exposure_cents"] for r in ok)
    bad_v = sum(r["_exposure_cents"] for r in bad)
    esc_v = sum(r["_exposure_cents"] for r in esc)

    classes = []
    for cls in sorted({r["exception_class"] or "" for r in esc}):
        g = [r for r in esc if (r["exception_class"] or "") == cls]
        v = sum(r["_exposure_cents"] for r in g)
        classes.append({
            "exception_class": cls,
            "rows": len(g),
            "exposure_cents": v,
            "pct_of_queue_rows": round(len(g) / len(esc) * 100, 4) if esc else 0.0,
            "pct_of_queue_value": round(v / esc_v * 100, 4) if esc_v else 0.0,
            "would_be_correct_cents": sum(r["_exposure_cents"] for r in g if r["_correct"]),
            "would_be_wrong_cents": sum(r["_exposure_cents"] for r in g if not r["_correct"]),
        })

    triggers = []
    for t in CTL.TRIGGERS:
        g = [r for r in esc if t in (r.get("triggers") or [])]
        v = sum(r["_exposure_cents"] for r in g)
        c = sum(1 for r in g if r["_correct"])
        pct = (c / len(g) * 100) if g else None
        # Verdict thresholds mirror controller.py's PER-TRIGGER VERDICTS table exactly.
        verdict = ("never fired" if not g else
                   "COSTING COVERAGE" if pct >= 50 else
                   "earning its keep" if pct <= 20 else "mixed")
        triggers.append({
            "trigger": t, "rows": len(g), "exposure_cents": v,
            "would_be_correct": c, "would_be_wrong": len(g) - c,
            "correct_pct": round(pct, 4) if pct is not None else None,
            "verdict": verdict,
        })

    return {
        "batch": name,
        "currency": currency,
        "records_processed": n,
        "auto_closed": len(auto),
        "escalated": len(esc),
        "auto_close_rate_pct": round(len(auto) / n * 100, 4),
        "escalation_rate_pct": round(len(esc) / n * 100, 4),
        "throughput_records_per_sec": (throughput or {}).get("records_per_sec"),
        "wall_clock_sec": (throughput or {}).get("wall_clock_sec"),
        "auto_close_precision_by_rows_pct": round(len(ok) / len(auto) * 100, 4) if auto else None,
        "auto_close_precision_by_value_pct": round(ok_v / auto_v * 100, 4) if auto_v else None,
        "overall_match_pct": round(len(correct_all) / n * 100, 4) if n else None,
        "overall_precision_pct": (round(len(correct_all) / n_predicted * 100, 4)
                                  if n_predicted else None),
        "overall_abstention_pct": round(len(abstained_all) / n * 100, 4) if n else None,
        "value": {
            "unit": "cents",
            "total_batch": tot_v,
            "auto_closed_total": auto_v,
            "auto_closed_correct": ok_v,
            "auto_closed_incorrect": bad_v,
            "in_exception_queue": esc_v,
        },
        "exposure_definition": {
            "wrongly_auto_closed": "full absolute B amount",
            "escalated": "full absolute B amount at risk pending review, including rows "
                         "whose correct answer is blank",
            "correctly_auto_closed": "zero exposure; counted only in the precision "
                                     "denominator",
            "note": "escalated exposure is workload; wrongly auto-closed exposure is "
                    "value already posted unchecked. They are not additive.",
        },
        "exception_classes": classes,
        "triggers": triggers,
    }


def compute_curve(recs, seeds=None, deciles=None):
    seeds = EXP.RANDOM_SEEDS if seeds is None else seeds
    deciles = EXP.DECILES if deciles is None else deciles
    esc = sorted((r["_exposure_cents"] for r in recs
                  if not r["_auto"]), reverse=True)
    if not esc:
        return {"points": [], "total_exposure_cents": 0, "queue_rows": 0}
    v = np.array(esc, dtype=np.int64)
    total = int(v.sum())
    n = len(v)
    ranked_cum = np.cumsum(v) / total

    rng = np.random.default_rng(0)
    rand = np.vstack([np.cumsum(v[rng.permutation(n)]) / total for _ in range(seeds)])

    pts = []
    for pct in deciles:
        k = max(1, int(round(n * pct / 100)))
        pts.append({
            "review_top_pct": pct,
            "rows_reviewed": k,
            "ranked_pct": round(float(ranked_cum[k - 1]) * 100, 4),
            "random_pct": round(float(rand[:, k - 1].mean()) * 100, 4),
            "random_sd": round(float(rand[:, k - 1].std()) * 100, 4),
        })
    return {"queue_rows": n, "total_exposure_cents": total,
            "random_seeds": seeds, "points": pts}


def one_line_evidence(rec):
    c = rec.get("candidates") or []
    bits = [f"{len(c)} cand"]
    if rec.get("top1_score") is not None:
        bits.append(f"top1={rec['top1_score']:.4f}")
    if rec.get("margin") is not None:
        bits.append(f"margin={rec['margin']:.4f}")
    bits.append("exact" if rec.get("exact_amount_top1") else "no-exact-amount")
    if rec.get("duplicate_reference_among_candidates"):
        bits.append("dup-ref")
    ak = rec.get("added_keys") or []
    if ak:
        ps = [a.get("probability") for a in ak if isinstance(a, dict)
              and a.get("probability") is not None]
        bits.append(f"+{len(ak)} added" + (f" (p max {max(ps):.3f})" if ps else ""))
    if not c:
        bits.append("empty pool")
    return "; ".join(bits)


def build_detail(rec, investigation, b_row=None, currency="USD"):
    ak = []
    for a in rec.get("added_keys") or []:
        if isinstance(a, dict):
            ak.append({"allocation_key": a.get("allocation_key"),
                       "probability": a.get("probability")})
        else:
            ak.append({"allocation_key": a, "probability": None})
    return {
        "b_id": rec["_b_id"],
        "batch": rec.get("batch"),
        "decision": rec["decision"],
        "exception_class": rec.get("exception_class"),
        "triggers": rec.get("triggers") or [],
        "transaction": {
            "amount_cents": rec["b_amount_cents"],
            "exposure_cents": rec["_exposure_cents"],
            "currency": (b_row or {}).get("B_currencyCode") or currency,
            "debit_or_credit": (b_row or {}).get("B_debitOrCredit"),
            "value_date": (b_row or {}).get("B_valueDate"),
            "import_date": (b_row or {}).get("B_importDate"),
            "references": (b_row or {}).get("B_transactionReferences"),
            "attributes": (b_row or {}).get("B_transactionAttributes"),
        },
        "candidate_pool_size": rec.get("pool_size"),
        "top1_a_id": rec.get("top1_a_id"),
        "top1_score": rec.get("top1_score"),
        "margin": rec.get("margin"),
        "exact_amount_top1": rec.get("exact_amount_top1"),
        "duplicate_reference_among_candidates": rec.get("duplicate_reference_among_candidates"),
        "candidates": [
            {"rank": c.get("rank"), "a_id": c.get("a_id"),
             "amount_cents": c.get("amount_cents"),
             "delta_from_b_cents": c.get("amount_delta_cents"),
             "similarity_score": c.get("score"),
             "exact_amount": c.get("exact_amount")}
            for c in (rec.get("candidates") or [])
        ],
        "added_keys": ak,
        "answer_keys": rec.get("answer_keys") or [],
        "investigation": investigation,
    }


# ----------------------------------------------------------------------------------
# Cross-check against exposure.py
# ----------------------------------------------------------------------------------
def check_gates(batch, summary, gates, failures):
    """Every gate figure must reconstruct from the batch's own row counts.

    driftgate.log is the source, but agreeing with itself proves nothing. These are the
    identities that tie its table back to the audit export.py computed independently:
    the no-gate state must equal this batch's real coverage and precision, and each
    gate's post-gate figures must follow from the rows it escalates."""
    if not gates:
        return
    k = batch["key"]
    recs = summary["records_processed"]
    auto = summary["auto_closed"]
    prec = summary["auto_close_precision_by_rows_pct"]
    correct = int(round(auto * prec / 100.0))

    def cmp(label, mine, theirs, tol=0.0):
        if mine is None or theirs is None:
            failures.append(f"{k}: gates {label} missing "
                            f"(driftgate={mine}, export={theirs})")
        elif abs(float(mine) - float(theirs)) > tol:
            failures.append(f"{k}: gates {label} MISMATCH "
                            f"driftgate.log={mine} export={theirs}")

    base = gates["states"][0]
    cmp("baseline records", base["records"], recs)
    cmp("baseline auto-closed", base["auto_closed"], auto)
    cmp("baseline correct", base["correct"], correct, tol=1)
    cmp("baseline coverage", base["coverage_pct"], summary["auto_close_rate_pct"],
        tol=0.001)
    cmp("baseline precision", base["precision_pct"], prec, tol=0.001)

    for st in gates["states"][1:]:
        g, esc = st["key"], st["escalates"]
        # a gate escalates exactly the rows it catches, right ones and wrong ones
        cmp(f"{g} escalates = wrong + correct",
            esc, st["wrong_caught"] + st["correct_given_up"])
        # and the wrong ones it catches are exactly the three drift outcomes
        cmp(f"{g} wrong = zero + nonzero + bounded", st["wrong_caught"],
            st["zero_drift_caught"] + st["nonzero_drift_caught"]
            + st["bounded_caught"])
        # the resulting operating point follows from the rows that remain
        left = auto - esc
        if left <= 0:
            failures.append(f"{k}: gates {g} escalates every auto-closed row")
            continue
        cmp(f"{g} coverage", st["coverage_pct"], left / recs * 100, tol=0.001)
        cmp(f"{g} precision", st["precision_pct"],
            (correct - st["correct_given_up"]) / left * 100, tol=0.002)
        if st["wrong_caught"]:
            cmp(f"{g} correct-per-wrong", st["correct_per_wrong"],
                st["correct_given_up"] / st["wrong_caught"], tol=0.05)
        # a gate cannot catch more invisible errors than exist
        if gates["invisible_total"] is not None \
                and st["zero_drift_caught"] > gates["invisible_total"]:
            failures.append(f"{k}: gates {g} catches {st['zero_drift_caught']} "
                            f"zero-drift rows but only {gates['invisible_total']} exist")


def check_operating_points(summary, ops, failures):
    """The shipped row of the sweep must be the batch's own headline figures.

    complete.log's threshold 0.5 row IS the configuration controller.py ran, so its
    overall match and precision have to equal the ones computed from the audit. If they
    ever diverge, the sweep is describing a different pipeline than the one that ran."""
    if not ops:
        return

    def cmp(label, mine, theirs, tol=0.0):
        if mine is None or theirs is None:
            failures.append(f"eval: operating points {label} missing")
        elif abs(float(mine) - float(theirs)) > tol:
            failures.append(f"eval: operating points {label} MISMATCH "
                            f"complete.log={mine} export={theirs}")

    ship = next((p for p in ops["points"] if p.get("shipped")), None)
    if ship is None:
        failures.append("eval: operating points carry no shipped threshold")
        return
    cmp("shipped overall match vs audit", ship["overall_match_pct"],
        summary["overall_match_pct"], tol=0.001)
    cmp("shipped overall precision vs audit", ship["overall_precision_pct"],
        summary["overall_precision_pct"], tol=0.001)
    for pt in ops["points"]:
        if pt["gained"] - pt["lost"] != pt["net"]:
            failures.append(f"eval: operating point {pt['threshold']} "
                            f"gained-lost != net ({pt['gained']}-{pt['lost']}"
                            f" != {pt['net']})")
    # completion off must be the zero-change row
    off = ops["completion_off"]
    if (off["gained"], off["lost"], off["net"], off["multi_match_pct"]) != (0, 0, 0, 0.0):
        failures.append("eval: completion-off row is not the null row "
                        f"({off['gained']}/{off['lost']}/{off['net']}/"
                        f"{off['multi_match_pct']})")


def cross_check(data_dir, batch, summary, curve, failures):
    """Run exposure.py's own implementation in-process and require agreement."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = EXP.load_batch(data_dir, batch["audit"], batch["solution"])
        ref = EXP.headline(df, batch["name"])

    def cmp(label, mine, theirs, tol=0):
        if theirs is None or mine is None:
            failures.append(f"{batch['key']}: {label} missing (mine={mine}, "
                            f"exposure={theirs})")
            return
        if abs(float(mine) - float(theirs)) > tol:
            failures.append(f"{batch['key']}: {label} MISMATCH "
                            f"mine={mine} exposure.py={theirs}")

    v = summary["value"]
    cmp("total batch value", v["total_batch"], ref["tot"])
    cmp("auto-closed value", v["auto_closed_total"], ref["auto"])
    cmp("value auto-closed correctly", v["auto_closed_correct"], ref["ok"])
    cmp("value auto-closed incorrectly", v["auto_closed_incorrect"], ref["bad"])
    cmp("queue value", v["in_exception_queue"], ref["esc"])

    # row counts and precisions, straight off exposure's own frame
    cmp("records processed", summary["records_processed"], len(df))
    cmp("auto-closed rows", summary["auto_closed"], int(df["auto_closed"].sum()))
    cmp("escalated rows", summary["escalated"], int((~df["auto_closed"]).sum()))
    auto = df[df["auto_closed"]]
    cmp("row-count precision", summary["auto_close_precision_by_rows_pct"],
        round(auto["correct"].mean() * 100, 4), tol=1e-4)
    # The blind-close figures are recomputed here from the audit; ctrl.log printed its
    # own. They must agree, or one of the two is measuring something else.
    ctrl = os.path.join(data_dir, "ctrl.log")
    if os.path.exists(ctrl) and summary["overall_match_pct"] is not None:
        txt = open(ctrl, encoding="utf-8", errors="replace").read()
        m = re.search(r"FINAL REPORT — " + re.escape(batch["name"]) +
                      r"\s*\n=+\n(.*?)(?=\n={10,}|\Z)", txt, re.S)
        if m:
            blk = m.group(1)
            om = re.search(r"overall match rate\s+([\d\.]+)%", blk)
            op = re.search(r"overall precision\s+([\d\.]+)%", blk)
            if om:
                cmp("overall match rate vs ctrl.log", summary["overall_match_pct"],
                    float(om.group(1)), tol=0.001)
            if op:
                cmp("overall precision vs ctrl.log", summary["overall_precision_pct"],
                    float(op.group(1)), tol=0.001)

    cmp("value-weighted precision", summary["auto_close_precision_by_value_pct"],
        round(ref["ok"] / ref["auto"] * 100, 4), tol=1e-4)

    # curve, using exposure's ordering and constants
    esc_v = df[~df["auto_closed"]].sort_values(
        "exposure_cents", ascending=False)["exposure_cents"].to_numpy()
    if len(esc_v):
        total = esc_v.sum()
        ranked_cum = np.cumsum(esc_v) / total
        rng = np.random.default_rng(0)
        rnd = np.vstack([np.cumsum(esc_v[rng.permutation(len(esc_v))]) / total
                         for _ in range(EXP.RANDOM_SEEDS)])
        for pt in curve["points"]:
            k = max(1, int(round(len(esc_v) * pt["review_top_pct"] / 100)))
            cmp(f"curve ranked @{pt['review_top_pct']}%", pt["ranked_pct"],
                round(float(ranked_cum[k - 1]) * 100, 4), tol=1e-4)
            cmp(f"curve random @{pt['review_top_pct']}%", pt["random_pct"],
                round(float(rnd[:, k - 1].mean()) * 100, 4), tol=1e-4)

    # queue value by class, against exposure's frame
    escdf = df[~df["auto_closed"]]
    for c in summary["exception_classes"]:
        g = escdf[escdf["exception_class"] == c["exception_class"]]
        cmp(f"class '{c['exception_class']}' rows", c["rows"], len(g))
        cmp(f"class '{c['exception_class']}' exposure", c["exposure_cents"],
            int(g["exposure_cents"].sum()))


# ----------------------------------------------------------------------------------
def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    return os.path.getsize(path)


def _main():
    ap = argparse.ArgumentParser(description="Export existing outputs as static JSON.")
    ap.add_argument("data_dir", nargs="?",
                    default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--synth", action="store_true",
                    help="also export the synthetic batch (off by default: it is 10,615 "
                         "detail files and a 2.2 MB queue, and it is not real data)")
    ap.add_argument("--only", choices=[b["key"] for b in BATCHES], default=None,
                    help="export exactly one batch")
    args = ap.parse_args()
    data_dir = args.data_dir
    out_root = os.path.join(data_dir, OUT_ROOT)

    if args.only:
        batches = [b for b in BATCHES if b["key"] == args.only]
    elif args.synth:
        batches = BATCHES
    else:
        batches = [b for b in BATCHES if b["key"] == "eval"]

    _rule("=")
    _log("EXPORT — static JSON for a front end")
    _rule("=")
    _log()
    _log("  Serialiser only: no decisions, no model calls, no analysis. Summary and")
    _log("  curve figures are recomputed from the audit files and then cross-checked")
    _log("  against exposure.py's own implementation. exposure.log is never parsed.")
    _log()

    if os.path.isdir(out_root):
        shutil.rmtree(out_root)     # stale detail files would silently outlive a rerun
    os.makedirs(out_root, exist_ok=True)

    investigations = load_investigations(data_dir)
    throughput = throughput_from_ctrl_log(data_dir)
    reference = reference_from_score_log(data_dir)
    regimes = regimes_from_findings_log(data_dir)
    op_points = operating_points_from_complete_log(data_dir)
    _log(f"  investigations available: {len(investigations)}")
    _log(f"  regimes from findings.log: "
         + (f"train {regimes['train']} / eval {regimes['eval']}" if regimes
            else "not available"))
    _log(f"  reference comparison from score.log: "
         + (f"{reference['source_file']} at {reference['match_rate_pct']}% / "
            f"{reference['match_precision_pct']}%" if reference else "not available"))
    _log(f"  throughput parsed from ctrl.log for: {sorted(throughput) or 'none'}")
    _log(f"  operating points from complete.log: "
         + (f"{len(op_points['points'])} thresholds, flat band "
            f"{op_points['flat_band']['from']}-{op_points['flat_band']['to']}"
            if op_points else "not available"))
    _log()

    _log(f"  exporting batches: {[b['key'] for b in batches]}"
         + ("" if len(batches) > 1 else "   (pass --synth to include synthetic)"))
    _log()

    files, failures = [], []
    for batch in batches:
        ap = os.path.join(data_dir, batch["audit"])
        if not os.path.exists(ap):
            _log(f"  {batch['audit']} missing — skipping {batch['name']}")
            continue

        recs = load_audit(data_dir, batch)
        b_side = load_b_side(data_dir, batch)
        summary = compute_summary(recs, batch["name"], throughput.get(batch["key"]),
                                  batch.get("currency", "USD"))
        # score.log scores the eval solution; there is no comparable file for a batch
        # we generated ourselves, so the synthetic summary simply carries none.
        if batch["key"] == "eval" and reference:
            summary["reference"] = reference
        # The regime finding is about BenchRec's own structure, measured on train and
        # eval. It says nothing about a batch we generated, so it rides with eval only.
        if batch["key"] == "eval" and regimes:
            summary["regimes"] = regimes
        # The gate comparison is measured per batch by driftgate.py, so both carry one.
        gates = gates_from_driftgate_log(data_dir, batch["name"])
        if gates:
            summary["gates"] = gates
        # The completion sweep was run on eval only; there is no synthetic counterpart.
        if batch["key"] == "eval" and op_points:
            summary["operating_points"] = op_points
        curve = compute_curve(recs)

        _log(f"  {batch['name']}: {len(recs):,} records, "
             f"{summary['escalated']:,} escalated")
        cross_check(data_dir, batch, summary, curve, failures)
        check_gates(batch, summary, gates, failures)
        if batch["key"] == "eval":
            check_operating_points(summary, op_points, failures)

        files.append((f"summary_{batch['key']}.json",
                      _write(os.path.join(out_root, f"summary_{batch['key']}.json"), summary)))
        files.append((f"curve_{batch['key']}.json",
                      _write(os.path.join(out_root, f"curve_{batch['key']}.json"), curve)))

        # queue, from the ranked CSV that exposure.py wrote
        rp = os.path.join(data_dir, batch["ranked"])
        by_id = {r["_b_id"]: r for r in recs}
        queue_rows = []
        if os.path.exists(rp):
            q = pd.read_csv(rp, dtype={"b_id": str})
            for r in q.itertuples():
                rec = by_id.get(str(r.b_id), {})
                queue_rows.append({
                    "rank": int(r.rank),
                    "b_id": str(r.b_id),
                    "exposure_cents": int(r.exposure_cents),
                    "exception_class": r.exception_class,
                    "triggers": (r.triggers or "").split("|") if r.triggers else [],
                    "evidence": one_line_evidence(rec),
                })
        else:
            _log(f"    {batch['ranked']} missing — queue built from audit order")
            esc = sorted([r for r in recs if not r["_auto"]],
                         key=lambda x: -x["_exposure_cents"])
            for i, rec in enumerate(esc, 1):
                queue_rows.append({
                    "rank": i, "b_id": rec["_b_id"],
                    "exposure_cents": rec["_exposure_cents"],
                    "exception_class": rec.get("exception_class"),
                    "triggers": rec.get("triggers") or [],
                    "evidence": one_line_evidence(rec),
                })
        # Paged. The index carries page 0 inline, so one small fetch is enough to
        # draw the top of the queue; the remaining pages are fetched behind it.
        pages = max(1, -(-len(queue_rows) // QUEUE_PAGE))
        head = {
            "batch": batch["name"],
            "rows": len(queue_rows),
            "total_exposure_cents": sum(r["exposure_cents"] for r in queue_rows),
            "page_size": QUEUE_PAGE,
            "pages": pages,
            "queue": queue_rows[:QUEUE_PAGE],
        }
        files.append((f"queue_{batch['key']}.json",
                      _write(os.path.join(out_root, f"queue_{batch['key']}.json"), head)))
        for i in range(1, pages):
            fn = f"queue_{batch['key']}_p{i}.json"
            files.append((fn, _write(os.path.join(out_root, fn),
                                     {"page": i,
                                      "queue": queue_rows[i * QUEUE_PAGE:(i + 1) * QUEUE_PAGE]})))
        _log(f"    queue: {len(queue_rows):,} rows over {pages} page(s) "
             f"of {QUEUE_PAGE:,}")

        # one detail file per escalated row
        ddir = os.path.join(out_root, "detail", batch["key"])
        n_det = n_inv = 0
        det_bytes = 0
        for rec in recs:
            if rec["_auto"]:
                continue
            inv = investigations.get(rec["_b_id"])
            if inv:
                n_inv += 1
            det_bytes += _write(os.path.join(ddir, f"{rec['_b_id']}.json"),
                                build_detail(rec, inv, b_side.get(rec["_b_id"]),
                                             batch.get("currency", "USD")))
            n_det += 1
        _log(f"    detail files: {n_det:,}  ({n_inv} with an LLM investigation)")
        files.append((f"detail/{batch['key']}/  ({n_det:,} files)", det_bytes))

    # ---- consistency gate -------------------------------------------------------
    _log()
    _rule("=")
    _log("CONSISTENCY CHECK vs exposure.py")
    _rule("=")
    _log()
    if failures:
        _log(f"  {len(failures)} DISAGREEMENT(S) — export is inconsistent with the analysis:")
        for f in failures[:40]:
            _log(f"    {f}")
        _log()
        _log("  Refusing to present these figures as correct. Fix the disagreement.")
        shutil.rmtree(out_root, ignore_errors=True)
        _log(f"  {OUT_ROOT} removed so nothing downstream reads contradictory numbers.")
        sys.exit(1)
    _log("  All recomputed figures match exposure.py exactly: totals, row counts, both")
    _log("  precisions, per-class rows and exposure, and every curve point (ranked and")
    _log("  random) at each review percentage.")

    # ---- size report ------------------------------------------------------------
    total = 0
    for _, b in files:
        total += b
    for dirpath, _, names in os.walk(out_root):
        pass
    n_files = sum(len(n) for _, _, n in os.walk(out_root))

    _log()
    _rule("=")
    _log("OUTPUT SIZE")
    _rule("=")
    _log()
    tab = pd.DataFrame([{"file": f, "bytes": b, "MB": round(b / 1e6, 3)}
                        for f, b in files])
    _log(tab.to_string(index=False))
    _log()
    _log(f"  files written : {n_files:,}")
    _log(f"  total size    : {total / 1e6:.2f} MB")
    _log()

    big = [(f, b) for f, b in files if b >= LARGE_FILE_BYTES and not f.startswith("detail/")]
    if big:
        _log(f"  SLOW-FETCH WARNING — {len(big)} single file(s) at or above "
             f"{LARGE_FILE_BYTES / 1e6:.1f} MB:")
        for f, b in big:
            _log(f"    {f:34s} {b / 1e6:6.2f} MB  — a browser will block on this; "
                 f"paginate or stream it")
    else:
        _log(f"  No single JSON file reaches {LARGE_FILE_BYTES / 1e6:.1f} MB.")
    det_dirs = [(f, b) for f, b in files if f.startswith("detail/")]
    if det_dirs:
        _log()
        _log("  Detail files are fetched one at a time by b_id, so their aggregate size")
        _log("  does not affect page load. Their file COUNT matters for deployment:")
        for f, b in det_dirs:
            _log(f"    {f:34s} {b / 1e6:6.2f} MB total")


if __name__ == "__main__":
    _main()
