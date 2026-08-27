"""
retrieve.py — retrieval of A_allocation keys for type-B transactions in
BenchRec_cash_v1.0_eval.csv, with two controlled experiments and an amount-tolerance
sweep.

Reference point (established previously):
    cosine + amount blocking @ 0.01  ->  single-key match 95.0760%, precision 98.4429%

EXPERIMENT A — drop the dead field.
    Build the B-side query from B_transactionAttributes ONLY, dropping
    B_transactionReferences (which shares a digit run with the A side on just 3.16%
    of true matches). Everything else held fixed.

EXPERIMENT B — structured numeric key.
    Extract digit runs of length 7-12 from B_transactionAttributes and the A-side
    text, measure how often true matches share one and how discriminative that is,
    then apply shared-run as (i) a hard filter and (ii) a strong score boost on top
    of cosine + amount.

AMOUNT TOLERANCE SWEEP — exact / 0.01 / 1.00 / 0.1% / 1% / 5% relative, each with
    its true-match ceiling, pool size, match rate and precision, so the tolerance is
    chosen deliberately rather than inherited.

Nothing is tuned against the eval labels. Label-derived figures are measurements.
The two free choices are stated and justified structurally:
  * run length 7 — the minimum of the requested 7-12 range. A shared run of length
    >= 7 always contains a shared 7-window, so testing at 7 is exactly the test
    "shares any run of length 7-12 (or longer)".
  * boost +1.0 — larger than any attainable cosine value, making the boost a strict
    lexicographic preference (shared-run candidates first, cosine as tiebreak). No
    magnitude was searched.

Run:  python retrieve.py [data_dir]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from score import score, _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


NGRAM_RANGE = (3, 5)
TOP_K = 5
DATE_WINDOW_DAYS = 7
QUERY_CHUNK = 256
RUN_LEN = 7                 # see module docstring
RUN_LENS_REPORT = range(7, 13)
BOOST = 1.0                 # see module docstring
REFERENCE_TOL = 0.01        # the inherited tolerance the experiments are run at

# Tolerances are expressed in INTEGER CENTS, and all amount comparisons are done in
# cents. Amounts here reach 6.6e9, where a float64 has ~1e-6 resolution, so a 0.01
# tolerance is not representable reliably in floating point: |a-b| for two values one
# cent apart can evaluate to 0.009999997. Comparing integer cents removes the
# ambiguity entirely and makes "exact" and "0.01" mean exactly what they say.
TOLERANCES = [
    ("exact", lambda cents: 0),
    ("0.01", lambda cents: 1),
    ("1.00", lambda cents: 100),
    ("0.1% rel", lambda cents: abs(cents) * 0.001),
    ("1% rel", lambda cents: abs(cents) * 0.01),
    ("5% rel", lambda cents: abs(cents) * 0.05),
]


def _log(msg=""):
    print(msg, flush=True)


def _rule(char="-", n=96):
    _log(char * n)


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def _load_sides(eval_path):
    ev = pd.read_csv(eval_path, dtype=str, keep_default_na=False)
    a = ev[ev["A_transactionType"] == "A"].copy().reset_index(drop=True)
    b = ev[ev["B_transactionType"] == "B"].copy().reset_index(drop=True)

    a["text"] = (a["A_transactionReferences"] + " " +
                 a["A_transactionAttributes"] + " " +
                 a["A_allocation"])
    # Two query constructions, identical except for the dropped field.
    b["text_full"] = b["B_transactionReferences"] + " " + b["B_transactionAttributes"]
    b["text_attr"] = b["B_transactionAttributes"]

    a["date"] = pd.to_datetime(a["A_valueDate"], errors="coerce", format="mixed")
    b["date"] = pd.to_datetime(b["B_valueDate"], errors="coerce", format="mixed")
    a["amt"] = pd.to_numeric(a["A_amount"], errors="coerce")
    b["amt"] = pd.to_numeric(b["B_amount"], errors="coerce")
    # Exact integer cents — see the note on TOLERANCES.
    a["cents"] = np.round(a["amt"] * 100).astype(np.int64)
    b["cents"] = np.round(b["amt"] * 100).astype(np.int64)
    return a, b


def _labels(solution_path):
    sol = pd.read_csv(solution_path, dtype=str, keep_default_na=False)
    return {b: _parse_alloc(t) for b, t in zip(sol["B_id"], sol["targetAllocation"])}


def _digit_windows(s, k):
    return {s[i:i + k] for i in range(len(s) - k + 1) if s[i:i + k].isdigit()}


# ----------------------------------------------------------------------------------
# Amount tolerance: ceilings and pool sizes
# ----------------------------------------------------------------------------------
def _tolerance_ceilings(a, b, labels, idx_by_key, window_days):
    a_amt, a_date = a["cents"].to_numpy(), a["date"].to_numpy()
    b_amt, b_date = b["cents"].to_numpy(), b["date"].to_numpy()
    w = np.timedelta64(window_days, "D")

    survive = {name: 0 for name, _ in TOLERANCES}
    n_single = 0
    for i, b_id in enumerate(b["B_id"].to_numpy()):
        gold = labels.get(b_id, set())
        if len(gold) != 1:
            continue
        n_single += 1
        ii = idx_by_key.get(next(iter(gold)), [])
        if not ii:
            continue
        ii = np.asarray(ii)
        in_win = np.abs(a_date[ii] - b_date[i]) <= w
        if not in_win.any():
            continue
        diffs = np.abs(a_amt[ii[in_win]] - b_amt[i])
        for name, tol_fn in TOLERANCES:
            if (diffs <= tol_fn(b_amt[i])).any():
                survive[name] += 1

    a_keys = (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy()
    b_keys = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    pool = {name: np.zeros(len(b), dtype=np.int64) for name, _ in TOLERANCES}
    pool_date = np.zeros(len(b), dtype=np.int64)

    gk = pd.Series(b_keys) + "||" + b["B_valueDate"].astype(str)
    for _, b_idx in gk.groupby(gk).indices.items():
        b_idx = np.asarray(b_idx)
        a_idx = np.where(a_keys == b_keys[b_idx[0]])[0]
        if len(a_idx) == 0:
            continue
        a_idx = a_idx[np.argsort(a_date[a_idx], kind="stable")]
        ds = a_date[a_idx]
        qd = b_date[b_idx[0]]
        cand = a_idx[np.searchsorted(ds, qd - w, "left"):np.searchsorted(ds, qd + w, "right")]
        pool_date[b_idx] = len(cand)
        if len(cand) == 0:
            continue
        amts = np.sort(a_amt[cand])
        for name, tol_fn in TOLERANCES:
            for j in b_idx:
                t = tol_fn(b_amt[j])
                pool[name][j] = (np.searchsorted(amts, b_amt[j] + t, "right") -
                                 np.searchsorted(amts, b_amt[j] - t, "left"))

    return survive, n_single, pool, pool_date


# ----------------------------------------------------------------------------------
# Experiment B measurements
# ----------------------------------------------------------------------------------
def _shared_run_analysis(a, b, labels, idx_by_key):
    _rule("=")
    _log("EXPERIMENT B (measurement) — SHARED DIGIT RUNS, LENGTH 7-12")
    _rule("=")
    _log()

    a_text = a["text"].to_numpy()

    hit_any = 0
    n = 0
    longest = []
    for b_id, b_attr in zip(b["B_id"].to_numpy(), b["B_transactionAttributes"].to_numpy()):
        gold = labels.get(b_id, set())
        if len(gold) != 1:
            continue
        ii = idx_by_key.get(next(iter(gold)), [])
        if not ii:
            continue
        n += 1
        best = 0
        for k in reversed(RUN_LENS_REPORT):
            wins = _digit_windows(b_attr, k)
            if wins and any(any(x in a_text[i] for x in wins) for i in ii):
                best = k
                break
        if best:
            hit_any += 1
            longest.append(best)

    _log(f"  True matches tested (single-key labels): {n:,}")
    _log(f"  Share at least one digit run of length 7-12: "
         f"{hit_any:,} / {n:,} = {hit_any / n * 100:.4f}%")
    _log(f"  Share none:                                 "
         f"{n - hit_any:,} / {n:,} = {(n - hit_any) / n * 100:.4f}%")
    _log()
    _log("  Distribution of the LONGEST shared run length (12 = 12 or more):")
    vc = pd.Series(longest).value_counts().sort_index()
    rows = [{"longest_shared_run": int(k), "true_matches": int(v),
             "pct_of_all_true_matches": round(v / n * 100, 4)} for k, v in vc.items()]
    _log(pd.DataFrame(rows).to_string(index=False))
    _log()
    _log("  Note: a shared run of length >= 7 necessarily contains a shared 7-window, so")
    _log("  the filter below tests length 7 — that is exactly the 'shares any run of")
    _log("  length 7-12' predicate, not a narrower one.")
    return hit_any / n


# ----------------------------------------------------------------------------------
# Main traversal — every configuration in one pass
# ----------------------------------------------------------------------------------
def run_all(a, b, window_days=DATE_WINDOW_DAYS, top_k=TOP_K, ngram_range=NGRAM_RANGE):
    _log()
    _rule("=")
    _log("RETRIEVAL PASS")
    _rule("=")
    _log()

    t0 = time.perf_counter()
    vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=2)
    # Vocabulary fit once on A text + the FULL B text, so Experiment A changes only the
    # query content, never the feature space. That keeps the comparison controlled.
    vec.fit(pd.concat([a["text"], b["text_full"]], ignore_index=True))
    x_a = vec.transform(a["text"])
    x_b_full = vec.transform(b["text_full"])
    x_b_attr = vec.transform(b["text_attr"])
    t_fit = time.perf_counter() - t0
    _log(f"  vocabulary {len(vec.vocabulary_):,} features; fit+transform {t_fit:.2f} s")
    _log(f"  mean query length: full {b['text_full'].str.len().mean():.1f} chars, "
         f"attributes-only {b['text_attr'].str.len().mean():.1f} chars")
    _log()

    a_keys = (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy()
    b_keys = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    a_alloc = a["A_allocation"].to_numpy()
    a_text = a["text"].to_numpy()
    a_date, b_date = a["date"].to_numpy(), b["date"].to_numpy()
    a_amt, b_amt = a["cents"].to_numpy(), b["cents"].to_numpy()
    b_attr = b["B_transactionAttributes"].to_numpy()
    n_b = len(b)

    names = (["cosine+amount", "expA_attr_only+amount",
              "expB_filter", "expB_boost", "expA+expB_boost"] +
             [f"sweep::{t}" for t, _ in TOLERANCES])
    R = {nm: {"alloc": np.array([""] * n_b, dtype=object),
              "score": np.zeros(n_b),
              "topk": [[] for _ in range(n_b)]} for nm in names}

    # Experiment B discriminativeness counters
    shared_counts, cand_counts = [], []
    mixed_n = argmax_shared_n = 0   # is the boost ever able to change the argmax?

    w = np.timedelta64(window_days, "D")
    t1 = time.perf_counter()
    gk = pd.Series(b_keys) + "||" + b["B_valueDate"].astype(str)

    def emit(nm, row, scores, cand_subset):
        """Rank only the candidates actually passed in. The amount block leaves ~14 of
        ~6,400, so working on the subset rather than masking the full array is what
        keeps this tractable."""
        if scores.size == 0:
            return
        k = min(top_k, scores.size)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        R[nm]["alloc"][row] = a_alloc[cand_subset[order[0]]]
        R[nm]["score"][row] = scores[order[0]]
        R[nm]["topk"][row] = list(a_alloc[cand_subset[order]])

    for _, b_idx in gk.groupby(gk).indices.items():
        b_idx = np.asarray(b_idx)
        a_idx = np.where(a_keys == b_keys[b_idx[0]])[0]
        if len(a_idx) == 0:
            continue
        a_idx = a_idx[np.argsort(a_date[a_idx], kind="stable")]
        ds = a_date[a_idx]
        qd = b_date[b_idx[0]]
        cand = a_idx[np.searchsorted(ds, qd - w, "left"):np.searchsorted(ds, qd + w, "right")]
        if len(cand) == 0:
            continue

        xa_c = x_a[cand]
        cand_amt = a_amt[cand]
        cand_text = a_text[cand]

        for start in range(0, len(b_idx), QUERY_CHUNK):
            chunk = b_idx[start:start + QUERY_CHUNK]
            s_full = (x_b_full[chunk] @ xa_c.T).toarray()
            s_attr = (x_b_attr[chunk] @ xa_c.T).toarray()
            absdiff = np.abs(cand_amt[None, :] - b_amt[chunk][:, None])

            masks = {}
            for tname, tol_fn in TOLERANCES:
                tv = np.array([tol_fn(b_amt[j]) for j in chunk], dtype=np.float64)
                masks[tname] = absdiff <= tv[:, None]
            ref_mask = masks[str(REFERENCE_TOL)] if str(REFERENCE_TOL) in masks else masks["0.01"]

            for i, row in enumerate(chunk):
                # ---- tolerance sweep, cosine on the full query ----
                for tname, _ in TOLERANCES:
                    surv = np.flatnonzero(masks[tname][i])
                    if surv.size:
                        emit(f"sweep::{tname}", row, s_full[i, surv], cand[surv])

                surv = np.flatnonzero(ref_mask[i])
                if surv.size == 0:
                    continue
                cand_s = cand[surv]

                emit("cosine+amount", row, s_full[i, surv], cand_s)
                emit("expA_attr_only+amount", row, s_attr[i, surv], cand_s)

                # ---- Experiment B: shared digit run among amount-surviving candidates ----
                qwins = _digit_windows(b_attr[row], RUN_LEN)
                if qwins:
                    shared = np.fromiter(
                        (any(x in cand_text[c] for x in qwins) for c in surv),
                        dtype=bool, count=surv.size)
                else:
                    shared = np.zeros(surv.size, dtype=bool)

                cand_counts.append(surv.size)
                shared_counts.append(int(shared.sum()))

                # Redundancy check: on queries where the boost COULD reorder (some but
                # not all surviving candidates share a run), does cosine already put a
                # shared-run candidate first? If it always does, the boost is a no-op by
                # construction rather than by coincidence.
                if 0 < shared.sum() < surv.size:
                    mixed_n += 1
                    if shared[np.argmax(s_full[i, surv])]:
                        argmax_shared_n += 1

                # hard filter: require a shared run, otherwise abstain
                if shared.any():
                    emit("expB_filter", row, s_full[i, surv][shared], cand_s[shared])

                # strong boost: shared-run candidates strictly outrank the rest
                emit("expB_boost", row, s_full[i, surv] + BOOST * shared, cand_s)
                emit("expA+expB_boost", row, s_attr[i, surv] + BOOST * shared, cand_s)

    t_retrieve = time.perf_counter() - t1
    _log(f"  retrieval, all configurations {t_retrieve:.2f} s")

    preds = {nm: pd.DataFrame({"B_id": b["B_id"].to_numpy(),
                               "targetAllocation": R[nm]["alloc"],
                               "score": R[nm]["score"]}) for nm in names}
    diag = {"shared_counts": np.array(shared_counts), "cand_counts": np.array(cand_counts),
            "mixed_n": mixed_n, "argmax_shared_n": argmax_shared_n}
    return preds, {nm: R[nm]["topk"] for nm in names}, diag, {
        "fit_seconds": t_fit, "retrieve_seconds": t_retrieve, "n_queries": n_b}


# ----------------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------------
def _rank_breakdown(preds, topk, labels):
    n = hit1 = hit5 = 0
    for b_id, top in zip(preds["B_id"].to_numpy(), topk):
        gold = labels.get(str(b_id), set())
        if len(gold) != 1:
            continue
        n += 1
        key = next(iter(gold))
        if top and top[0] == key:
            hit1 += 1
        if key in (top or []):
            hit5 += 1
    return n, hit1, hit5


def _row(name, preds, topk, labels, solution_path):
    res = score(preds[["B_id", "targetAllocation"]], solution_path)
    sk = res["by_label_type"]["single_key"]
    n, hit1, hit5 = _rank_breakdown(preds, topk, labels)
    return {
        "config": name,
        "single_match_%": round(sk["match_rate"] * 100, 4),
        "single_prec_%": None if sk["match_precision"] is None else round(sk["match_precision"] * 100, 4),
        "top1_%": round(hit1 / n * 100, 4),
        "not_in_top5_%": round((n - hit5) / n * 100, 4),
        "predicted": sk["predicted"],
        "abstain_%": round(res["abstention_rate"] * 100, 4),
        "overall_match_%": round(res["match_rate"] * 100, 4),
    }


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(data_dir, "BenchRec_cash_v1.0_eval.csv")
    solution_path = os.path.join(data_dir, "BenchRec_cash_v1.0_solution.csv")

    wall0 = time.perf_counter()
    a, b = _load_sides(eval_path)
    labels = _labels(solution_path)
    idx_by_key = {}
    for i, k in enumerate(a["A_allocation"].to_numpy()):
        idx_by_key.setdefault(k, []).append(i)

    survive, n_single, pool, pool_date = _tolerance_ceilings(
        a, b, labels, idx_by_key, DATE_WINDOW_DAYS)
    _shared_run_analysis(a, b, labels, idx_by_key)

    preds, topks, diag, timing = run_all(a, b)

    # ---------------- Experiment B discriminativeness ----------------
    _log()
    _rule("=")
    _log("EXPERIMENT B (measurement) — HOW DISCRIMINATIVE IS A SHARED RUN?")
    _rule("=")
    _log()
    sc, cc = diag["shared_counts"], diag["cand_counts"]
    _log(f"  Queries measured (amount block @ {REFERENCE_TOL} non-empty): {len(cc):,}")
    _log(f"  Candidates surviving amount blocking   mean {cc.mean():.2f}   median {np.median(cc):.0f}")
    _log(f"  Of those, sharing a digit run          mean {sc.mean():.2f}   median {np.median(sc):.0f}")
    _log(f"  Share of surviving candidates that share a run: {sc.sum() / cc.sum() * 100:.4f}%")
    _log()
    _log(f"  Queries where NO surviving candidate shares a run: "
         f"{int((sc == 0).sum()):,} ({(sc == 0).mean() * 100:.4f}%)")
    _log(f"  Queries where exactly one does:                    "
         f"{int((sc == 1).sum()):,} ({(sc == 1).mean() * 100:.4f}%)")
    _log(f"  Queries where more than one does:                  "
         f"{int((sc > 1).sum()):,} ({(sc > 1).mean() * 100:.4f}%)")
    _log()
    frac = sc.sum() / cc.sum()
    _log(f"  Amount blocking has already cut the pool to ~{cc.mean():.0f}, so the question is")
    _log("  whether a shared run narrows it further.")
    _log()
    if frac > 0.30:
        _log(f"  It does NOT narrow it much: {frac * 100:.2f}% of amount-surviving candidates")
        _log("  share a run with the query. A predicate that keeps well over a third of the")
        _log("  pool is a weak discriminator at this stage — most of its power was already")
        _log("  spent by the amount block.")
    else:
        _log(f"  It narrows it meaningfully: only {frac * 100:.2f}% of amount-surviving")
        _log("  candidates share a run with the query.")
    _log()
    _log(f"  Critically, on {(sc == 0).mean() * 100:.2f}% of queries NO surviving candidate")
    _log("  shares a run at all. As a hard filter that forces an abstention on every one of")
    _log("  them, which caps match rate accordingly — the boost form avoids that by falling")
    _log("  back to cosine.")
    _log()
    mx, ax = diag["mixed_n"], diag["argmax_shared_n"]
    _log("  Redundancy check — can the boost change anything at all?")
    _log(f"      queries where boost COULD reorder (some but not all candidates share): "
         f"{mx:,}")
    _log(f"      of those, cosine's top-1 ALREADY shares a run: {ax:,} "
         f"({ax / max(mx, 1) * 100:.4f}%)")
    if mx and ax == mx:
        _log()
        _log("      That is 100%. The boost cannot change a single prediction, and not by")
        _log("      coincidence: the shared digit run IS what the character n-grams are")
        _log("      scoring on. Cosine has already absorbed this signal, so adding it")
        _log("      explicitly is redundant.")

    # ---------------- Tolerance sweep ----------------
    _log()
    _rule("=")
    _log("AMOUNT TOLERANCE SWEEP — choose deliberately")
    _rule("=")
    _log()
    rows = []
    for tname, _ in TOLERANCES:
        nm = f"sweep::{tname}"
        r = _row(tname, preds[nm], topks[nm], labels, solution_path)
        rows.append({
            "tolerance": tname,
            "ceiling_%": round(survive[tname] / n_single * 100, 4),
            "true_dropped_%": round(100 - survive[tname] / n_single * 100, 4),
            "pool_mean": round(float(pool[tname].mean()), 2),
            "pool_median": float(np.median(pool[tname])),
            "single_match_%": r["single_match_%"],
            "single_prec_%": r["single_prec_%"],
            "abstain_%": r["abstain_%"],
            "gap_to_ceiling": round(survive[tname] / n_single * 100 - r["single_match_%"], 4),
        })
    sweep = pd.DataFrame(rows)
    _log(f"  Date-window-only pool for reference: {pool_date.mean():.1f} candidates/query")
    _log()
    _log(sweep.to_string(index=False))
    _log()
    best_match = sweep.loc[sweep["single_match_%"].idxmax()]
    best_prec = sweep.loc[sweep["single_prec_%"].idxmax()]
    _log(f"  Highest match rate:  {best_match['tolerance']} at "
         f"{best_match['single_match_%']:.4f}% (ceiling {best_match['ceiling_%']:.4f}%)")
    _log(f"  Highest precision:   {best_prec['tolerance']} at "
         f"{best_prec['single_prec_%']:.4f}%")
    _log()
    _log("  'gap_to_ceiling' is how much the similarity function loses BELOW what the")
    _log("  tolerance admits. A small gap means the tolerance, not the text scoring, is")
    _log("  the binding constraint.")

    # ---------------- Experiments A and B ----------------
    _log()
    _rule("=")
    _log("EXPERIMENTS A AND B — all at amount tolerance 0.01, everything else fixed")
    _rule("=")
    _log()
    exp_rows = [_row(nm, preds[nm], topks[nm], labels, solution_path)
                for nm in ["cosine+amount", "expA_attr_only+amount",
                           "expB_filter", "expB_boost", "expA+expB_boost"]]
    exp = pd.DataFrame(exp_rows)
    _log(exp.to_string(index=False))

    base = exp_rows[0]
    _log()
    _rule("=")
    _log("WHICH EXPERIMENT HELPED, AND BY HOW MUCH")
    _rule("=")
    _log()
    _log(f"  Reference — cosine + amount@0.01:")
    _log(f"      top-1 {base['top1_%']:.4f}%   not-in-top-5 {base['not_in_top5_%']:.4f}%   "
         f"precision {base['single_prec_%']:.4f}%")
    _log()

    for r, title in [(exp_rows[1], "EXPERIMENT A — B_transactionAttributes only"),
                     (exp_rows[2], "EXPERIMENT B — shared run as a HARD FILTER"),
                     (exp_rows[3], "EXPERIMENT B — shared run as a STRONG BOOST"),
                     (exp_rows[4], "A + B together (boost form)")]:
        d1 = r["top1_%"] - base["top1_%"]
        d5 = r["not_in_top5_%"] - base["not_in_top5_%"]
        dp = r["single_prec_%"] - base["single_prec_%"]
        dm = r["single_match_%"] - base["single_match_%"]
        if abs(d1) < 0.01 and abs(dp) < 0.01:
            verdict = "NO EFFECT — indistinguishable from the reference"
        elif dm > 0:
            verdict = "HELPED on match rate"
        else:
            verdict = "DID NOT HELP on match rate"
        _log(f"  {title}")
        _log(f"      top-1          {r['top1_%']:.4f}%  ({d1:+.4f} pts)")
        _log(f"      not-in-top-5   {r['not_in_top5_%']:.4f}%  ({d5:+.4f} pts)")
        _log(f"      match rate     {r['single_match_%']:.4f}%  ({dm:+.4f} pts)")
        _log(f"      precision      {r['single_prec_%']:.4f}%  ({dp:+.4f} pts)")
        _log(f"      abstention     {r['abstain_%']:.4f}%")
        _log(f"      -> {verdict}")
        _log()

    wall = time.perf_counter() - wall0
    _rule("=")
    _log("RUNTIME")
    _rule("=")
    _log()
    _log(f"  fit + transform        {timing['fit_seconds']:>8.2f} s")
    _log(f"  retrieval (all configs){timing['retrieve_seconds']:>8.2f} s")
    _log(f"  wall clock             {wall:>8.2f} s")
    _log(f"  queries                {timing['n_queries']:>8,}")
    _log(f"  records/sec            {timing['n_queries'] / timing['retrieve_seconds']:>8,.1f}")


if __name__ == "__main__":
    _main()
