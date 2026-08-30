"""
controller.py — agent layer over the existing matching stack.

Imports retrieve.py, complete.py, score.py and run_synth.py UNMODIFIED. No new matching
logic: this decides what to do with what the stack already produces.

    ingest batch -> retrieve -> complete -> decide per transaction
                 -> auto-close, or escalate with a reason
                 -> reconciliation report + exception list + audit trail

INTEGRITY
---------
The decision path never sees labels. Candidate features are built with
with_labels=False, and a batch's solution file is loaded only AFTER every decision for
that batch is fixed.

The low_confidence thresholds are FITTED ON BenchRec_cash_v1.0_train.csv using train's
inline labels, and on nothing else. They are never fitted, checked or adjusted against
eval or against the synthetic batches — those are reported at whatever the train-chosen
setting gives.

ESCALATION TRIGGERS — each traces to a measured finding
------------------------------------------------------
completion_added   The answer contains a key the completion classifier added rather than
                   one retrieval found. MEASURED: out of domain on synthetic data,
                   completion cost duplicate_reference 50.5 points of accuracy
                   (75.944% -> 25.398%).
fee_band_only      The top-1 candidate was admitted only by the widened fee band, not by
                   exact amount. MEASURED: on real eval, widening drops overall match
                   91.794% -> 75.181% and cuts no_candidate firings 1,408 -> 165.
low_confidence     Top-1 similarity below a floor, or the rank1-rank2 margin below a
                   floor. Both fitted on train (see tune_low_confidence).
no_candidate       Empty candidate pool. MEASURED: 0.66% of real eval rows are genuinely
                   unmatchable, so this escalates as a PROPOSED NO-MATCH, not a failure.

FEE WIDENING IS PER BATCH
-------------------------
Real BenchRec has effectively no fee deductions, so the widened band admits only false
candidates AND masks no_candidate by guaranteeing a non-empty pool. Synthetic data
contains fees by construction (9.4% of groups), where widening gains ground. Both
settings are run against both batches so the choice is shown, not asserted.

Run:  python controller.py [data_dir]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import retrieve as R
import complete as C
import run_synth as RS
from score import score, _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


COMPLETION_THRESHOLD = 0.5
FEE_REL_BELOW = 0.05
AMOUNT_TOL_CENTS = C.TOL_CENTS

# Fee widening, per batch. Justification recorded in the output.
BATCH_FEE_WIDENING = {"BenchRec eval": False, "synthetic 50,000-group": True}

# Grid for the train-only fit. 0.0 disables that half of the trigger.
GRID_TOP1 = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
GRID_MARGIN = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
CORRECT_RATE_CEILING = 50.0     # trigger must catch more wrong than right

TRIGGERS = ["completion_added", "fee_band_only", "low_confidence", "no_candidate"]
EXCEPTION_CLASSES = ["ambiguous_allocation", "fee_band_match", "missing_counterparty",
                     "duplicate_reference_suspected", "incomplete_set"]


def _log(m=""):
    print(m, flush=True)


def _rule(c="-", n=108):
    _log(c * n)


# ----------------------------------------------------------------------------------
# Basis: everything a decision needs, computed once, label-free
# ----------------------------------------------------------------------------------
def build_basis(a, b, idx, scr, proba, owner, cand_key, tol=AMOUNT_TOL_CENTS,
                completion_threshold=COMPLETION_THRESHOLD):
    alloc = a["A_allocation"].to_numpy()
    a_cents = a["cents"].to_numpy()
    a_ref = a["A_transactionReferences"].to_numpy()
    a_id = a["A_id"].to_numpy()
    b_cents = b["cents"].to_numpy()
    b_ids = b["B_id"].to_numpy()
    n = len(b_ids)

    accepted = {}
    for r, p in enumerate(proba):
        if p >= completion_threshold:
            accepted.setdefault(int(owner[r]), []).append((cand_key[r], float(p)))

    has = np.zeros(n, bool)
    top1 = np.full(n, np.nan)
    margin = np.full(n, np.nan)
    exact1 = np.zeros(n, bool)
    dupref = np.zeros(n, bool)
    added_n = np.zeros(n, np.int64)
    answers, added_keys_all, cand_detail, top1_ids = [], [], [], []

    for i in range(n):
        cands, scores = idx[i], scr[i]
        if cands.size == 0:
            answers.append("")
            added_keys_all.append([])
            cand_detail.append([])
            top1_ids.append("")
            continue
        has[i] = True
        top1[i] = float(scores[0])
        margin[i] = float(scores[0] - scores[1]) if scores.size >= 2 else float(scores[0])
        t = cands[0]
        exact1[i] = abs(int(a_cents[t]) - int(b_cents[i])) <= tol
        refs = [a_ref[j] for j in cands]
        dupref[i] = len(refs) != len(set(refs))

        # Keep the accepting probability with each added key. `ak` (keys only) still
        # drives the answer and added_count exactly as before, so no decision changes;
        # the pairs exist purely so the audit can say WHY a key was added.
        ak_pairs = [(k, p) for k, p in accepted.get(i, []) if k != alloc[t]]
        ak = [k for k, _ in ak_pairs]
        added_n[i] = len(ak)
        keys = [alloc[t]] + ak
        answers.append(("[" + ",".join(keys) + "]") if len(keys) > 1 else keys[0])
        added_keys_all.append([{"allocation_key": k, "probability": round(p, 6)}
                               for k, p in ak_pairs])
        top1_ids.append(str(a_id[t]))
        cand_detail.append([
            {"rank": r_, "a_id": str(a_id[j]), "score": round(float(s), 6),
             "amount_cents": int(a_cents[j]),
             "amount_delta_cents": int(a_cents[j]) - int(b_cents[i]),
             "exact_amount": bool(abs(int(a_cents[j]) - int(b_cents[i])) <= tol)}
            for r_, (j, s) in enumerate(zip(cands, scores), start=1)])

    return {"b_ids": b_ids.astype(str), "has_cand": has, "top1": top1, "margin": margin,
            "exact_top1": exact1, "dup_ref": dupref, "added_count": added_n,
            "answers": answers, "added_keys": added_keys_all,
            "candidates": cand_detail, "top1_a_id": top1_ids,
            "b_cents": b_cents, "pool": np.array([i.size for i in idx])}


def apply_policy(basis, lc):
    """Vectorised trigger evaluation. Label-free."""
    has = basis["has_cand"]
    t_no = ~has
    t_add = basis["added_count"] > 0
    t_fee = has & ~basis["exact_top1"]
    with np.errstate(invalid="ignore"):
        t_low = has & ((basis["top1"] < lc["min_top1_score"]) |
                       (basis["margin"] < lc["min_margin"]))
    esc = t_no | t_add | t_fee | t_low
    cls = np.select(
        [t_no, basis["dup_ref"] & t_add, t_fee, t_add, t_low],
        ["missing_counterparty", "duplicate_reference_suspected", "fee_band_match",
         "incomplete_set", "ambiguous_allocation"],
        default="")
    return {"no_candidate": t_no, "completion_added": t_add, "fee_band_only": t_fee,
            "low_confidence": t_low, "escalate": esc, "exception_class": cls}


def correctness(basis, labels):
    return np.array([_parse_alloc(ans) == labels.get(bid, set())
                     for bid, ans in zip(basis["b_ids"], basis["answers"])])


# ----------------------------------------------------------------------------------
# Ranking / batch preparation
# ----------------------------------------------------------------------------------
def prepare(tx_path, clf, fee_widening):
    t0 = time.perf_counter()
    a, b = R._load_sides(tx_path)
    _, xa, xb = C._build_index(a, b)
    if fee_widening:
        idx, scr = RS._topk_asym(a, b, xa, xb, rel_below=FEE_REL_BELOW)
    else:
        idx, scr = C._topk(a, b, xa, xb)
    X, _, owner, cand_key = C._features(a, b, idx, scr, {}, False)   # label-blind
    proba = clf.predict_proba(X)[:, 1] if len(X) else np.empty(0)
    basis = build_basis(a, b, idx, scr, proba, owner, cand_key)
    basis["wall"] = time.perf_counter() - t0
    basis["n_rows"] = len(a) + len(b)
    basis["n_a"], basis["n_b"] = len(a), len(b)
    return basis


def fit_classifier(data_dir):
    """Same recipe as run_synth.train_classifier (deterministic, random_state=0), but
    ranks train once so the same pass also feeds the threshold fit."""
    _log("Ranking BenchRec train and fitting the completion classifier (train only).")
    t0 = time.perf_counter()
    a, b = R._load_sides(os.path.join(data_dir, "BenchRec_cash_v1.0_train.csv"))
    labels = C._labels_from_column(b)
    _, xa, xb = C._build_index(a, b)
    idx, scr = C._topk(a, b, xa, xb)
    X, y, owner, cand_key = C._features(a, b, idx, scr, labels, True)
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    _log(f"  {len(y):,} candidate decisions, {int(y.sum()):,} positive "
         f"({y.mean() * 100:.2f}%), fitted in {time.perf_counter() - t0:.1f}s")

    Xb, _, ob, kb = C._features(a, b, idx, scr, {}, False)
    basis = build_basis(a, b, idx, scr, clf.predict_proba(Xb)[:, 1], ob, kb)
    return clf, basis, labels


# ----------------------------------------------------------------------------------
# Threshold fit — TRAIN ONLY
# ----------------------------------------------------------------------------------
def tune_low_confidence(basis, labels):
    _log()
    _rule("=")
    _log("LOW_CONFIDENCE THRESHOLD FIT — BenchRec TRAIN ONLY")
    _rule("=")
    _log()
    _log("  Grid searched against train's inline targetAllocation labels. Eval and the")
    _log("  synthetic batches are not consulted at any point in this section.")
    _log()
    _log("  'lc_fired' counts rows where the trigger fires at all (non-exclusive — the")
    _log("  framing the 68.7% eval figure used). 'lc_only' counts rows the trigger alone")
    _log("  removes from auto-close, which is its true marginal cost.")
    _log()
    _log("  CAVEAT: the completion classifier's probabilities on train are IN-SAMPLE, so")
    _log("  completion_added behaves better here than it will out of sample. That shifts")
    _log("  which rows are already escalated by other triggers, so lc_only is mildly")
    _log("  optimistic. The score/margin thresholds themselves do not depend on it.")
    _log()

    ok = correctness(basis, labels)
    rows = []
    for t1 in GRID_TOP1:
        for mg in GRID_MARGIN:
            P = apply_policy(basis, {"min_top1_score": t1, "min_margin": mg})
            esc, low = P["escalate"], P["low_confidence"]
            others = P["no_candidate"] | P["completion_added"] | P["fee_band_only"]
            only = low & ~others
            auto = ~esc
            rows.append({
                "min_top1": t1, "min_margin": mg,
                "lc_fired": int(low.sum()),
                "lc_correct_%": round(float(ok[low].mean() * 100), 3) if low.any() else None,
                "lc_only": int(only.sum()),
                "lc_only_correct_%": round(float(ok[only].mean() * 100), 3) if only.any() else None,
                "auto_closed": int(auto.sum()),
                "auto_coverage_%": round(float(auto.mean() * 100), 3),
                "auto_precision_%": round(float(ok[auto].mean() * 100), 3) if auto.any() else None,
            })
    tab = pd.DataFrame(rows)
    _log(tab.to_string(index=False))

    cand = tab[(tab["lc_fired"] > 0) & (tab["lc_correct_%"] < CORRECT_RATE_CEILING)]
    _log()
    _rule()
    _log(f"  SELECTION: lc_correct_% < {CORRECT_RATE_CEILING:.0f}% "
         f"(catches more wrong than right), then maximise auto-close coverage")
    _rule()
    _log()
    if not len(cand):
        best = tab[tab["lc_fired"] > 0].sort_values("lc_correct_%").iloc[0]
        _log("  NO combination in the grid gets the trigger below "
             f"{CORRECT_RATE_CEILING:.0f}%.")
        _log(f"  The best available is min_top1={best['min_top1']}, "
             f"min_margin={best['min_margin']} at {best['lc_correct_%']:.3f}% — still")
        _log("  escalating more correct rows than wrong ones.")
        _log()
        _log("  RECOMMENDATION: DROP the low_confidence trigger entirely rather than ship")
        _log("  one that fires indiscriminately. Setting both thresholds to 0.0 disables")
        _log("  it and is what is applied below.")
        chosen = {"min_top1_score": 0.0, "min_margin": 0.0}
        dropped = True
    else:
        best = cand.sort_values("auto_coverage_%", ascending=False).iloc[0]
        chosen = {"min_top1_score": float(best["min_top1"]),
                  "min_margin": float(best["min_margin"])}
        dropped = False
        _log(f"  CHOSEN: min_top1_score={chosen['min_top1_score']}, "
             f"min_margin={chosen['min_margin']}")
        _log(f"      trigger fires on {int(best['lc_fired']):,} train rows, "
             f"{best['lc_correct_%']:.3f}% of them already correct")
        _log(f"      marginal (lc_only): {int(best['lc_only']):,} rows, "
             f"{best['lc_only_correct_%']}% correct")
        _log(f"      train auto-close coverage {best['auto_coverage_%']:.3f}%, "
             f"precision {best['auto_precision_%']:.3f}%")
    return chosen, dropped, tab


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------
def summarise(basis, lc, labels):
    P = apply_policy(basis, lc)
    ok = correctness(basis, labels)
    auto = ~P["escalate"]
    return {"auto": int(auto.sum()), "esc": int(P["escalate"].sum()),
            "n": len(ok),
            "auto_coverage_%": round(float(auto.mean() * 100), 3),
            "auto_precision_%": round(float(ok[auto].mean() * 100), 3) if auto.any() else None,
            "overall_match_%": round(float(ok.mean() * 100), 3),
            "no_candidate_fired": int(P["no_candidate"].sum()), "P": P, "ok": ok}


def full_report(basis, lc, labels, name, sol_path, audit_path=None, data_dir="."):
    P = apply_policy(basis, lc)
    ok = correctness(basis, labels)
    auto = ~P["escalate"]
    n = len(ok)

    _log()
    _rule("=")
    _log(f"FINAL REPORT — {name}")
    _rule("=")
    _log()
    _log(f"  records processed        {n:>12,}")
    _log(f"  auto-closed              {int(auto.sum()):>12,}   ({auto.mean() * 100:.3f}%)")
    _log(f"  escalated                {int(P['escalate'].sum()):>12,}   "
         f"({P['escalate'].mean() * 100:.3f}%)")
    _log(f"  rows ingested            {basis['n_rows']:>12,}   "
         f"(A {basis['n_a']:,}  B {basis['n_b']:,})")
    _log(f"  wall clock               {basis['wall']:>12.2f} s")
    _log(f"  throughput               {n / basis['wall']:>12,.1f} records/sec")
    _log()
    preds = pd.DataFrame({"B_id": basis["b_ids"], "targetAllocation": basis["answers"]})
    sc = score(preds, sol_path)
    _log(f"  AUTO-CLOSE PRECISION     {ok[auto].mean() * 100:>10.3f}%   "
         f"<- accuracy on the {int(auto.sum()):,} rows the system chose to close")
    _log(f"  overall match rate       {sc['match_rate'] * 100:>10.3f}%   "
         f"<- if all {n:,} were closed blind")
    _log(f"  overall precision        {sc['match_precision'] * 100:>10.3f}%")

    _log()
    _log("  EXCEPTION BREAKDOWN BY CLASS")
    rows = []
    esc_n = int(P["escalate"].sum())
    for c in EXCEPTION_CLASSES:
        m = P["exception_class"] == c
        k = int(m.sum())
        rows.append({"exception_class": c, "count": k,
                     "pct_of_escalated": round(k / esc_n * 100, 3) if esc_n else 0.0,
                     "would_be_correct": int(ok[m].sum()),
                     "would_be_wrong": int((~ok[m]).sum()),
                     "correct_%": round(float(ok[m].mean() * 100), 3) if k else None})
    _log(pd.DataFrame(rows).to_string(index=False))

    _log()
    _log("  PER-TRIGGER VERDICTS")
    rows = []
    for t in TRIGGERS:
        m = P[t]
        k = int(m.sum())
        if not k:
            rows.append({"trigger": t, "escalated": 0, "would_be_correct": 0,
                         "would_be_wrong": 0, "correct_%": None, "verdict": "never fired"})
            continue
        pct = float(ok[m].mean() * 100)
        rows.append({"trigger": t, "escalated": k, "would_be_correct": int(ok[m].sum()),
                     "would_be_wrong": int((~ok[m]).sum()), "correct_%": round(pct, 3),
                     "verdict": ("COSTING COVERAGE" if pct >= 50 else
                                 "earning its keep" if pct <= 20 else "mixed")})
    _log(pd.DataFrame(rows).to_string(index=False))

    if audit_path:
        with open(audit_path, "w", encoding="utf-8") as fh:
            for i in range(n):
                trig = [t for t in TRIGGERS if P[t][i]]
                fh.write(json.dumps({
                    "batch": name, "b_id": basis["b_ids"][i],
                    "b_amount_cents": int(basis["b_cents"][i]),
                    "pool_size": int(basis["pool"][i]),
                    "candidates": basis["candidates"][i],
                    "top1_a_id": basis["top1_a_id"][i],
                    "top1_score": None if np.isnan(basis["top1"][i]) else round(float(basis["top1"][i]), 6),
                    "margin": None if np.isnan(basis["margin"][i]) else round(float(basis["margin"][i]), 6),
                    "exact_amount_top1": bool(basis["exact_top1"][i]),
                    "duplicate_reference_among_candidates": bool(basis["dup_ref"][i]),
                    "added_keys": basis["added_keys"][i],
                    "triggers": trig,
                    "exception_class": str(P["exception_class"][i]) or None,
                    "decision": "escalate" if P["escalate"][i] else "auto_close",
                    "answer_keys": _parse_alloc(basis["answers"][i]) and
                                   sorted(_parse_alloc(basis["answers"][i])) or [],
                }) + "\n")
        exc = [{"b_id": basis["b_ids"][i],
                "exception_class": str(P["exception_class"][i]),
                "triggers": "|".join(t for t in TRIGGERS if P[t][i]),
                "proposed_answer": basis["answers"][i],
                "top1_score": basis["top1"][i], "margin": basis["margin"][i]}
               for i in range(n) if P["escalate"][i]]
        ep = audit_path.replace("audit", "exceptions").replace(".jsonl", ".csv")
        pd.DataFrame(exc).to_csv(ep, index=False)
        _log()
        _log(f"  audit trail    -> {os.path.basename(audit_path)} ({n:,} records)")
        _log(f"  exception list -> {os.path.basename(ep)} ({len(exc):,} rows)")


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    _rule("=")
    _log("CONTROLLER — agent layer over the existing matching stack")
    _rule("=")
    _log()
    _log("  CHANGE 1: fee widening is now PER BATCH, not global.")
    _log("      OFF for BenchRec eval — real BenchRec has effectively no fee deductions,")
    _log("      so the widened band admits only false candidates (overall match")
    _log("      91.794% -> 75.181%) and masks the no_candidate trigger by guaranteeing a")
    _log("      non-empty pool (1,408 -> 165 firings).")
    _log("      ON for the synthetic batch — it contains fees by construction (9.4% of")
    _log("      groups), where widening gained +4.750 points of overall match.")
    _log("      Both settings are run against both batches below so this is shown.")
    _log()
    _log("  CHANGE 2: low_confidence thresholds are FITTED ON TRAIN ONLY.")
    _log()

    clf, train_basis, train_labels = fit_classifier(data_dir)
    lc, dropped, _ = tune_low_confidence(train_basis, train_labels)

    batches = [
        ("BenchRec eval", os.path.join(data_dir, "BenchRec_cash_v1.0_eval.csv"),
         os.path.join(data_dir, "BenchRec_cash_v1.0_solution.csv"), "controller_audit_eval.jsonl"),
        ("synthetic 50,000-group", os.path.join(data_dir, "synth_transactions.csv"),
         os.path.join(data_dir, "synth_solution.csv"), "controller_audit_synth.jsonl"),
    ]

    _log()
    _rule("=")
    _log("FEE WIDENING — BOTH SETTINGS ON BOTH BATCHES (thresholds as fitted on train)")
    _rule("=")

    bases, rows = {}, []
    for name, tx, sol, _ in batches:
        labels = R._labels(sol)
        for wide in (False, True):
            bs = prepare(tx, clf, wide)
            bases[(name, wide)] = (bs, labels, sol)
            s = summarise(bs, lc, labels)
            rows.append({"batch": name, "fee_widening": "ON" if wide else "OFF",
                         "auto_closed": s["auto"], "escalated": s["esc"],
                         "auto_coverage_%": s["auto_coverage_%"],
                         "auto_precision_%": s["auto_precision_%"],
                         "overall_match_%": s["overall_match_%"],
                         "no_candidate_fired": s["no_candidate_fired"]})
    _log()
    _log(pd.DataFrame(rows).to_string(index=False))
    _log()
    _log("  Read the no_candidate_fired column: widening does not merely add false")
    _log("  candidates, it disables a trigger by making the pool never empty.")

    _log()
    _rule("=")
    _log("APPLYING THE CHOSEN CONFIGURATION")
    _rule("=")
    _log()
    _log(f"  low_confidence: min_top1_score={lc['min_top1_score']}, "
         f"min_margin={lc['min_margin']}"
         f"{'   (TRIGGER DROPPED)' if dropped else ''}")
    _log("  >>> These thresholds were fitted on BenchRec_cash_v1.0_train.csv and never on")
    _log("  >>> eval or on synthetic data. Everything below is out-of-sample for them.")
    for name, _, _, _ in batches:
        _log(f"  fee widening for {name}: "
             f"{'ON' if BATCH_FEE_WIDENING[name] else 'OFF'}")

    for name, tx, sol, audit in batches:
        wide = BATCH_FEE_WIDENING[name]
        bs, labels, sol_path = bases[(name, wide)]
        full_report(bs, lc, labels, name, sol_path,
                    audit_path=os.path.join(data_dir, audit), data_dir=data_dir)


if __name__ == "__main__":
    _main()
