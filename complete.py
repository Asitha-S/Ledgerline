"""
complete.py — set completion on top of the existing retriever.

retrieve.py is NOT modified and NOT imported for its experiment harness; only its
loaders, label parser and constants are reused. The candidate ordering below
reproduces the deployed retriever's recipe (char 3-5 TF-IDF cosine, +/-7 day value
date window, amount block in integer cents) because feature extraction needs the
candidate ROW indices, which the retriever only returns as allocation keys. A
self-check asserts that this reproduction agrees with retrieve.py's own top-1.

The problem: retrieval emits exactly one key. 1,779 eval rows carry multi-key labels
and score zero under exact-set-equality. This module decides, for each of the
remaining top-5 candidates, whether it belongs in the allocation set.

Structure follows the brief and is gated:
    Step 1  measurement on train's multi-key groups (no modelling)
    Step 2  recall ceiling from top-5 — bounds everything downstream
    Step 3  classifier, ONLY if step 2's ceiling is workable

Run:  python complete.py [data_dir]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

import retrieve as R
from score import score, _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOP_K = 5
TOL_CENTS = 1            # 0.01, the retriever's reference tolerance
RUN_LEN = R.RUN_LEN      # 7
WINDOW = R.DATE_WINDOW_DAYS
CEILING_THRESHOLD = 15.0  # below this, step 3 is not worth building — declared up front


def _log(m=""):
    print(m, flush=True)


def _rule(c="-", n=96):
    _log(c * n)


# ----------------------------------------------------------------------------------
# Candidate ranking — mirrors the deployed retriever
# ----------------------------------------------------------------------------------
def _build_index(a, b):
    vec = TfidfVectorizer(analyzer="char", ngram_range=R.NGRAM_RANGE, min_df=2)
    vec.fit(pd.concat([a["text"], b["text_full"]], ignore_index=True))
    return vec, vec.transform(a["text"]), vec.transform(b["text_full"])


def _topk(a, b, xa, xb, tol_cents=TOL_CENTS, top_k=TOP_K):
    """Returns per-B-row (candidate row indices, cosine scores), best first, drawn
    from the amount-blocked candidate set."""
    ak = (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy()
    bk = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    ad, bd = a["date"].to_numpy(), b["date"].to_numpy()
    ac, bc = a["cents"].to_numpy(), b["cents"].to_numpy()
    w = np.timedelta64(WINDOW, "D")

    out_idx = [np.empty(0, dtype=np.int64)] * len(b)
    out_scr = [np.empty(0)] * len(b)

    gk = pd.Series(bk) + "||" + b["B_valueDate"].astype(str)
    for _, bi in gk.groupby(gk).indices.items():
        bi = np.asarray(bi)
        ai = np.where(ak == bk[bi[0]])[0]
        if len(ai) == 0:
            continue
        ai = ai[np.argsort(ad[ai], kind="stable")]
        ds = ad[ai]
        qd = bd[bi[0]]
        cand = ai[np.searchsorted(ds, qd - w, "left"):np.searchsorted(ds, qd + w, "right")]
        if len(cand) == 0:
            continue
        ca = ac[cand]
        for start in range(0, len(bi), 256):
            chunk = bi[start:start + 256]
            S = (xb[chunk] @ xa[cand].T).toarray()
            for i, row in enumerate(chunk):
                surv = np.flatnonzero(np.abs(ca - bc[row]) <= tol_cents)
                if surv.size == 0:
                    continue
                s = S[i, surv]
                k = min(top_k, s.size)
                part = np.argpartition(-s, k - 1)[:k]
                order = part[np.argsort(-s[part])]
                out_idx[row] = cand[surv[order]]
                out_scr[row] = s[order]
    return out_idx, out_scr


def _labels_from_column(b):
    return {bid: _parse_alloc(t)
            for bid, t in zip(b["B_id"].to_numpy(), b["targetAllocation"].to_numpy())}


def _dwins(s, k=RUN_LEN):
    return {s[i:i + k] for i in range(len(s) - k + 1) if s[i:i + k].isdigit()}


# ----------------------------------------------------------------------------------
# STEP 1 — measurement on train's multi-key groups
# ----------------------------------------------------------------------------------
def step1(a, b, labels, idx, scr):
    _rule("=")
    _log("STEP 1 — WHAT DISTINGUISHES THE ADDITIONAL KEYS' A ROWS (train, multi-key)")
    _rule("=")
    _log()

    alloc = a["A_allocation"].to_numpy()
    a_cents = a["cents"].to_numpy()
    a_text = a["text"].to_numpy()
    a_date = a["date"].to_numpy()
    b_cents = b["cents"].to_numpy()
    b_date = b["date"].to_numpy()
    b_ids = b["B_id"].to_numpy()

    # A rows grouped by allocation key
    by_key = {}
    for i, k in enumerate(alloc):
        by_key.setdefault(k, []).append(i)

    n_multi = top1_in_gold = 0
    same_date = tot_extra = 0
    run_with_top1 = 0
    sum_exact = top1_alone_exact = sum_closer = 0
    rank2_in_gold = rank2_extra = 0
    n_with_rank2 = 0

    for i, bid in enumerate(b_ids):
        gold = labels.get(bid, set())
        if len(gold) < 2:
            continue
        n_multi += 1
        if idx[i].size == 0:
            continue
        t1_key = alloc[idx[i][0]]
        if t1_key not in gold:
            continue
        top1_in_gold += 1
        extra = gold - {t1_key}

        t1_rows = by_key.get(t1_key, [])
        t1_date = a_date[t1_rows[0]] if t1_rows else None
        t1_text = a_text[t1_rows[0]] if t1_rows else ""
        t1_wins = _dwins(t1_text)

        # (a) value date, (c) shared digit runs with the top-1 A row
        for k in extra:
            rows = by_key.get(k, [])
            if not rows:
                continue
            tot_extra += 1
            if t1_date is not None and a_date[rows[0]] == t1_date:
                same_date += 1
            if t1_wins and any(x in a_text[r] for r in rows for x in t1_wins):
                run_with_top1 += 1

        # (b) do the amounts sum toward B?
        all_rows = [r for k in gold for r in by_key.get(k, [])]
        if all_rows:
            s_all = int(a_cents[all_rows].sum())
            s_t1 = int(a_cents[t1_rows].sum()) if t1_rows else 0
            tgt = int(b_cents[i])
            if abs(s_all - tgt) <= TOL_CENTS:
                sum_exact += 1
            if abs(s_t1 - tgt) <= TOL_CENTS:
                top1_alone_exact += 1
            if abs(s_all - tgt) < abs(s_t1 - tgt):
                sum_closer += 1

        # (d) is the second-best cosine candidate one of the additional keys?
        keys_ranked = [alloc[j] for j in idx[i]]
        second = next((k for k in keys_ranked[1:] if k != t1_key), None)
        if second is not None:
            n_with_rank2 += 1
            if second in gold:
                rank2_in_gold += 1
            if second in extra:
                rank2_extra += 1

    _log(f"  train B rows with a multi-key label:        {n_multi:,}")
    _log(f"  of those, retrieved top-1 is IN the gold set: {top1_in_gold:,} "
         f"({top1_in_gold / max(n_multi, 1) * 100:.2f}%)")
    _log(f"  additional gold keys examined:              {tot_extra:,}")
    _log()
    _log("  (a) VALUE DATE")
    _log(f"      additional key's A rows share the top-1 A row's value date: "
         f"{same_date / max(tot_extra, 1) * 100:.2f}%")
    _log()
    _log("  (b) AMOUNTS — do they sum toward B?")
    _log(f"      top-1's A rows alone equal B's amount exactly:  "
         f"{top1_alone_exact / max(top1_in_gold, 1) * 100:.2f}%")
    _log(f"      ALL gold keys' A rows sum to B's amount exactly:"
         f" {sum_exact / max(top1_in_gold, 1) * 100:.2f}%")
    _log(f"      adding the extra rows moves the sum CLOSER to B:"
         f" {sum_closer / max(top1_in_gold, 1) * 100:.2f}%")
    _log()
    _log("  (c) SHARED DIGIT RUNS between additional A rows and the top-1 A row")
    _log(f"      share a run of length >= {RUN_LEN}: "
         f"{run_with_top1 / max(tot_extra, 1) * 100:.2f}%")
    _log()
    _log("  (d) IS THE SECOND-BEST COSINE CANDIDATE A TRUE ADDITIONAL KEY?")
    _log(f"      rows with a distinct rank-2 candidate: {n_with_rank2:,}")
    _log(f"      rank-2 key is in the gold set:         "
         f"{rank2_in_gold / max(n_with_rank2, 1) * 100:.2f}%")
    _log(f"      rank-2 key is a genuinely ADDITIONAL key: "
         f"{rank2_extra / max(n_with_rank2, 1) * 100:.2f}%")
    _log()

    interp = []
    if same_date / max(tot_extra, 1) > 0.9:
        interp.append("value date does not separate them — the additional keys sit on the "
                      "same date as the top-1, so the date window cannot help")
    if sum_exact / max(top1_in_gold, 1) < 0.5:
        interp.append("amounts do NOT reliably sum to B, so a subset-sum rule is not "
                      "available")
    if rank2_extra / max(n_with_rank2, 1) > 0.5:
        interp.append("the rank-2 candidate IS usually a true additional key, which is the "
                      "signal a completion step can exploit")
    else:
        interp.append("the rank-2 candidate is usually NOT a true additional key, so "
                      "completion has to be selective rather than greedy")
    for s in interp:
        _log(f"  -> {s}")


# ----------------------------------------------------------------------------------
# STEP 2 — recall ceiling
# ----------------------------------------------------------------------------------
def step2(a, b, labels, idx, tag):
    _log()
    _rule("=")
    _log(f"STEP 2 — RECALL CEILING FROM TOP-{TOP_K} ({tag})")
    _rule("=")
    _log()

    alloc = a["A_allocation"].to_numpy()
    b_ids = b["B_id"].to_numpy()

    n_multi = formable = top1_present = 0
    n_single = single_formable = 0
    for i, bid in enumerate(b_ids):
        gold = labels.get(bid, set())
        keys = set(alloc[idx[i]]) if idx[i].size else set()
        if len(gold) >= 2:
            n_multi += 1
            if gold <= keys:
                formable += 1
            if gold & keys:
                top1_present += 1
        elif len(gold) == 1:
            n_single += 1
            if gold <= keys:
                single_formable += 1

    ceil_multi = formable / max(n_multi, 1) * 100
    _log(f"  multi-key rows:                              {n_multi:,}")
    _log(f"  gold set FULLY contained in top-{TOP_K}:            {formable:,} "
         f"({ceil_multi:.2f}%)   <-- the ceiling")
    _log(f"  at least one gold key in top-{TOP_K}:              {top1_present:,} "
         f"({top1_present / max(n_multi, 1) * 100:.2f}%)")
    _log()
    _log(f"  single-key rows:                             {n_single:,}")
    _log(f"  gold key present in top-{TOP_K}:                   {single_formable:,} "
         f"({single_formable / max(n_single, 1) * 100:.2f}%)")
    _log()
    _log(f"  A perfect subset-picker over the top-{TOP_K} could raise multi-key match rate")
    _log(f"  from 0% to at most {ceil_multi:.2f}%. Since multi-key rows are "
         f"{n_multi / len(b_ids) * 100:.2f}% of the file,")
    _log(f"  that is worth at most {ceil_multi / 100 * n_multi / len(b_ids) * 100:.2f} "
         f"points of overall match rate.")
    return ceil_multi, n_multi


# ----------------------------------------------------------------------------------
# STEP 3 — the classifier
# ----------------------------------------------------------------------------------
def _features(a, b, idx, scr, labels, with_labels):
    """One row per (query, distinct candidate key at rank >= 2)."""
    alloc = a["A_allocation"].to_numpy()
    a_cents = a["cents"].to_numpy()
    a_text = a["text"].to_numpy()
    a_date = a["date"].to_numpy()
    b_cents = b["cents"].to_numpy()
    b_date = b["date"].to_numpy()
    b_ids = b["B_id"].to_numpy()
    b_attr = b["B_transactionAttributes"].to_numpy()

    X, y, owner, cand_key = [], [], [], []
    for i in range(len(b_ids)):
        if idx[i].size < 2:
            continue
        t1 = idx[i][0]
        t1_key, t1_score = alloc[t1], scr[i][0]
        t1_wins = _dwins(a_text[t1])
        q_wins = _dwins(b_attr[i])
        keys_in_top = [alloc[j] for j in idx[i]]
        n_distinct = len(set(keys_in_top))
        gold = labels.get(b_ids[i], set()) if with_labels else set()

        seen = {t1_key}
        for rank, (j, s) in enumerate(zip(idx[i][1:], scr[i][1:]), start=2):
            k = alloc[j]
            if k in seen:
                continue
            seen.add(k)
            X.append([
                rank,
                s,
                t1_score - s,
                s / t1_score if t1_score > 0 else 0.0,
                np.log1p(abs(int(a_cents[j]) - int(b_cents[i]))),
                np.log1p(abs(int(a_cents[t1]) - int(b_cents[i]))),
                np.log1p(abs(int(a_cents[j]) + int(a_cents[t1]) - int(b_cents[i]))),
                float(a_cents[j] == a_cents[t1]),
                float((a_date[j] - b_date[i]) / np.timedelta64(1, "D")),
                float(a_date[j] == a_date[t1]),
                float(bool(q_wins) and any(x in a_text[j] for x in q_wins)),
                float(bool(t1_wins) and any(x in a_text[j] for x in t1_wins)),
                n_distinct,
                float(idx[i].size),
                np.log1p(len(a_text[j])),
            ])
            y.append(1 if k in gold else 0)
            owner.append(i)
            cand_key.append(k)
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int64),
            np.array(owner, dtype=np.int64), np.array(cand_key, dtype=object))


FEATURE_NAMES = ["rank", "score", "score_gap_to_top1", "score_ratio", "log_amt_diff",
                 "log_top1_amt_diff", "log_pair_sum_diff", "same_amt_as_top1",
                 "date_diff_days", "same_date_as_top1", "shares_run_with_query",
                 "shares_run_with_top1", "n_distinct_keys_top5", "n_candidates",
                 "log_text_len"]


def step3(a_tr, b_tr, lab_tr, idx_tr, scr_tr,
          a_ev, b_ev, lab_ev, idx_ev, scr_ev, solution_path):
    _log()
    _rule("=")
    _log("STEP 3 — CLASSIFIER: DOES THIS CANDIDATE BELONG IN THE SET?")
    _rule("=")
    _log()

    Xtr, ytr, _, _ = _features(a_tr, b_tr, idx_tr, scr_tr, lab_tr, True)
    _log(f"  training rows (candidate decisions): {len(ytr):,}")
    _log(f"  positives (candidate is in gold):    {int(ytr.sum()):,} "
         f"({ytr.mean() * 100:.4f}%)")
    _log(f"  features: {', '.join(FEATURE_NAMES)}")
    _log()
    _log("  Trained on TRAIN only, default hyperparameters, decision threshold 0.5.")
    _log("  No eval-label tuning.")

    clf = HistGradientBoostingClassifier(random_state=0)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    _log(f"  fit in {time.perf_counter() - t0:.2f} s")

    Xev, yev, owner, cand_key = _features(a_ev, b_ev, idx_ev, scr_ev, lab_ev, True)
    proba = clf.predict_proba(Xev)[:, 1] if len(Xev) else np.empty(0)
    accept = proba >= 0.5
    _log()
    _log(f"  eval candidate decisions: {len(yev):,}   accepted: {int(accept.sum()):,}")
    if len(yev):
        tp = int((accept & (yev == 1)).sum())
        fp = int((accept & (yev == 0)).sum())
        _log(f"  candidate-level precision: {tp / max(tp + fp, 1) * 100:.2f}%   "
             f"recall: {tp / max(int(yev.sum()), 1) * 100:.2f}%")

    alloc_ev = a_ev["A_allocation"].to_numpy()
    b_ids = b_ev["B_id"].to_numpy()
    p_off = _assemble(b_ids, idx_ev, alloc_ev, proba, owner, cand_key, threshold=2.0)
    return p_off, proba, owner, cand_key, alloc_ev, b_ids


def _assemble(b_ids, idx_ev, alloc_ev, proba, owner, cand_key, threshold):
    """Build a predictions frame at a given acceptance threshold. A threshold above 1.0
    accepts nothing, which is exactly completion-off."""
    extra = {}
    for r, p in enumerate(proba):
        if p >= threshold:
            extra.setdefault(int(owner[r]), []).append(cand_key[r])

    out = []
    for i in range(len(b_ids)):
        if idx_ev[i].size == 0:
            out.append("")
            continue
        t1 = alloc_ev[idx_ev[i][0]]
        ex = [k for k in extra.get(i, []) if k != t1]
        out.append("[" + ",".join([t1] + ex) + "]" if ex else t1)
    return pd.DataFrame({"B_id": b_ids, "targetAllocation": out})


def _correct_mask(pred_df, labels):
    """Row-level correctness under the same exact-set-equality rule score() applies."""
    return np.array([_parse_alloc(t) == labels.get(str(bid), set())
                     for bid, t in zip(pred_df["B_id"], pred_df["targetAllocation"])])


def sweep(p_off, proba, owner, cand_key, alloc_ev, b_ids, idx_ev, lab_ev, solution_path):
    _log()
    _rule("=")
    _log("ACCEPTANCE THRESHOLD SWEEP")
    _rule("=")
    _log()
    _log("  Rows gained/lost are row-level against completion-off: 'gained' = wrong with")
    _log("  completion off and correct with it on, 'lost' = the reverse. Completion can")
    _log("  only ever break a single-key answer, never fix one, since adding a key makes")
    _log("  the set size >= 2.")
    _log()

    base = score(p_off, solution_path)
    base_single_prec = base["by_label_type"]["single_key"]["match_precision"] * 100
    off_ok = _correct_mask(p_off, lab_ev)

    rows = [{
        "threshold": "OFF",
        "overall_match_%": round(base["match_rate"] * 100, 4),
        "overall_prec_%": round(base["match_precision"] * 100, 4),
        "single_match_%": round(base["by_label_type"]["single_key"]["match_rate"] * 100, 4),
        "single_prec_%": round(base_single_prec, 4),
        "multi_match_%": round(base["by_label_type"]["multi_key"]["match_rate"] * 100, 4),
        "gained": 0, "lost": 0, "net": 0,
    }]

    for t in np.arange(0.50, 0.9501, 0.05):
        p = _assemble(b_ids, idx_ev, alloc_ev, proba, owner, cand_key, float(t))
        res = score(p, solution_path)
        on_ok = _correct_mask(p, lab_ev)
        gained = int((~off_ok & on_ok).sum())
        lost = int((off_ok & ~on_ok).sum())
        sk = res["by_label_type"]["single_key"]
        rows.append({
            "threshold": round(float(t), 2),
            "overall_match_%": round(res["match_rate"] * 100, 4),
            "overall_prec_%": round(res["match_precision"] * 100, 4),
            "single_match_%": round(sk["match_rate"] * 100, 4),
            "single_prec_%": round(sk["match_precision"] * 100, 4),
            "multi_match_%": round(res["by_label_type"]["multi_key"]["match_rate"] * 100, 4),
            "gained": gained, "lost": lost, "net": gained - lost,
        })

    tab = pd.DataFrame(rows)
    _log(tab.to_string(index=False))

    _log()
    _rule()
    _log("  WHERE DOES SINGLE-KEY PRECISION RETURN TO THE COMPLETION-OFF BASELINE?")
    _rule()
    _log()
    _log(f"  completion-off single-key precision: {base_single_prec:.4f}%")
    _log(f"  target: >= 98.4%")
    _log()
    ok = tab[(tab["threshold"] != "OFF") & (tab["single_prec_%"] >= 98.4)]
    if not len(ok):
        _log("  NO threshold in 0.50-0.95 restores single-key precision to 98.4%.")
        best = tab[tab["threshold"] != "OFF"].sort_values("single_prec_%").iloc[-1]
        _log(f"  The closest is threshold {best['threshold']:.2f} at "
             f"{best['single_prec_%']:.4f}%, still "
             f"{98.4 - best['single_prec_%']:.4f} pts short, with multi-key match "
             f"{best['multi_match_%']:.4f}%.")
    else:
        first = ok.iloc[0]
        _log(f"  First threshold at or above 98.4%: {first['threshold']:.2f}")
        _log(f"      single-key precision   {first['single_prec_%']:.4f}%  "
             f"(baseline {base_single_prec:.4f}%)")
        _log(f"      MULTI-KEY MATCH RATE   {first['multi_match_%']:.4f}%   <- what survives")
        _log(f"      rows gained {int(first['gained']):,}  lost {int(first['lost']):,}  "
             f"net {int(first['net']):+,}")
        _log(f"      overall match {first['overall_match_%']:.4f}%   "
             f"overall precision {first['overall_prec_%']:.4f}%")
    _log()
    _log("  Not picking one — the table is above.")
    return tab


def _report(p_off, p_on, solution_path):
    _log()
    _rule("=")
    _log("RESULTS — COMPLETION OFF vs ON")
    _rule("=")

    rows = []
    for tag, p in [("completion OFF", p_off), ("completion ON", p_on)]:
        res = score(p, solution_path)
        r = {"config": tag,
             "overall_match_%": round(res["match_rate"] * 100, 4),
             "overall_prec_%": None if res["match_precision"] is None
                               else round(res["match_precision"] * 100, 4)}
        for lt in ("single_key", "multi_key", "blank"):
            m = res["by_label_type"][lt]
            r[f"{lt}_match_%"] = round(m["match_rate"] * 100, 4)
            r[f"{lt}_prec_%"] = (None if m["match_precision"] is None
                                 else round(m["match_precision"] * 100, 4))
            r[f"{lt}_correct"] = m["correct"]
        rows.append(r)

    df = pd.DataFrame(rows)
    _log()
    _log(df[["config", "overall_match_%", "overall_prec_%",
             "single_key_match_%", "single_key_prec_%",
             "multi_key_match_%", "blank_match_%"]].to_string(index=False))
    _log()
    _log("  Correct counts by label type:")
    _log(df[["config", "single_key_correct", "multi_key_correct",
             "blank_correct"]].to_string(index=False))

    a, bb = rows[0], rows[1]
    _log()
    _rule()
    _log("  NET EFFECT OF COMPLETION")
    _rule()
    _log()
    d_multi = bb["multi_key_correct"] - a["multi_key_correct"]
    d_single = bb["single_key_correct"] - a["single_key_correct"]
    d_overall = bb["overall_match_%"] - a["overall_match_%"]
    d_prec = bb["overall_prec_%"] - a["overall_prec_%"]
    _log(f"      multi-key rows gained:  {d_multi:+,}")
    _log(f"      single-key rows LOST:   {d_single:+,}   "
         f"(a wrongly added key breaks a correct answer)")
    _log(f"      net rows:               {d_multi + d_single:+,}")
    _log(f"      overall match rate:     {a['overall_match_%']:.4f}% -> "
         f"{bb['overall_match_%']:.4f}%  ({d_overall:+.4f} pts)")
    _log(f"      overall precision:      {a['overall_prec_%']:.4f}% -> "
         f"{bb['overall_prec_%']:.4f}%  ({d_prec:+.4f} pts)")
    _log(f"      single-key precision:   {a['single_key_prec_%']:.4f}% -> "
         f"{bb['single_key_prec_%']:.4f}%  "
         f"({bb['single_key_prec_%'] - a['single_key_prec_%']:+.4f} pts)")
    _log()
    if d_multi + d_single > 0:
        _log("      -> Completion is a NET GAIN.")
    elif d_multi + d_single == 0:
        _log("      -> Completion is NET NEUTRAL: it gains and loses the same number.")
    else:
        _log("      -> Completion is a NET LOSS. It breaks more single-key answers than")
        _log("         it fixes multi-key ones. Leave it off.")
    return rows


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, "BenchRec_cash_v1.0_train.csv")
    eval_path = os.path.join(data_dir, "BenchRec_cash_v1.0_eval.csv")
    solution_path = os.path.join(data_dir, "BenchRec_cash_v1.0_solution.csv")

    t0 = time.perf_counter()
    _log("Loading and ranking candidates (retriever recipe reproduced, retrieve.py untouched)")
    a_ev, b_ev = R._load_sides(eval_path)
    lab_ev = R._labels(solution_path)
    vec_e, xa_e, xb_e = _build_index(a_ev, b_ev)
    idx_ev, scr_ev = _topk(a_ev, b_ev, xa_e, xb_e)

    # Self-check: does the reproduction agree with retrieve.py's own top-1?
    alloc_ev = a_ev["A_allocation"].to_numpy()
    mine = np.array([alloc_ev[i[0]] if i.size else "" for i in idx_ev], dtype=object)
    preds_r, _, _, _ = R.run_all(a_ev, b_ev)
    theirs = preds_r["cosine+amount"]["targetAllocation"].to_numpy()
    agree = (mine == theirs).mean()
    _log(f"  self-check vs retrieve.py top-1: {agree * 100:.4f}% identical")
    if agree < 0.999:
        _log("  WARNING: reproduction diverges from the retriever; results below are not")
        _log("  comparable to the retriever's reported numbers.")

    a_tr, b_tr = R._load_sides(train_path)
    lab_tr = _labels_from_column(b_tr)
    vec_t, xa_t, xb_t = _build_index(a_tr, b_tr)
    idx_tr, scr_tr = _topk(a_tr, b_tr, xa_t, xb_t)
    _log(f"  ranked train ({len(b_tr):,} B rows) and eval ({len(b_ev):,} B rows) "
         f"in {time.perf_counter() - t0:.1f}s")
    _log()

    step1(a_tr, b_tr, lab_tr, idx_tr, scr_tr)
    step2(a_tr, b_tr, lab_tr, idx_tr, "train")
    ceiling, n_multi = step2(a_ev, b_ev, lab_ev, idx_ev, "eval — this is the binding one")

    _log()
    _rule("=")
    _log("GATE — IS THE CEILING WORKABLE?")
    _rule("=")
    _log()
    _log(f"  eval multi-key ceiling: {ceiling:.2f}%   (gate: {CEILING_THRESHOLD:.0f}%)")
    if ceiling < CEILING_THRESHOLD:
        _log()
        _log("  BELOW THE GATE. Stopping before step 3 rather than building on it.")
        _log(f"  Even a perfect subset-picker could only fix {ceiling:.2f}% of multi-key")
        _log("  rows, and any classifier will fall well short of perfect while risking")
        _log("  single-key answers that are currently correct.")
        return
    _log("  Above the gate — proceeding to step 3.")

    p_off, proba, owner, cand_key, alloc_ev, b_ids = step3(
        a_tr, b_tr, lab_tr, idx_tr, scr_tr,
        a_ev, b_ev, lab_ev, idx_ev, scr_ev, solution_path)

    p_on = _assemble(b_ids, idx_ev, alloc_ev, proba, owner, cand_key, 0.5)
    _report(p_off, p_on, solution_path)
    sweep(p_off, proba, owner, cand_key, alloc_ev, b_ids, idx_ev, lab_ev, solution_path)

    out = os.path.join(data_dir, "complete_predictions.csv")
    p_on.to_csv(out, index=False)
    _log()
    _log(f"  predictions (completion ON, threshold 0.5) written to {out}")
    _log(f"  total wall clock {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    _main()
