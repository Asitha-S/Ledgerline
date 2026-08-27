"""
run_synth.py — run the full pipeline (retriever + completion classifier @ 0.5) over the
synthetic batches and score with the existing scorer.

retrieve.py, complete.py and score.py are imported UNMODIFIED. Everything here is new
code; the only additions are (a) an asymmetric amount block for the fee experiment and
(b) the per-class reporting the synthetic manifest makes possible.

OUT-OF-DOMAIN NOTICE
--------------------
The completion classifier is fitted on BenchRec_cash_v1.0_train.csv and is NOT retrained,
refitted or calibrated on synthetic data at any point. It is applied to the synthetic
batches out of domain. That transfer is part of what this run measures, so its results
should be read as a domain-shift test, not as an in-domain performance figure.

Run:  python run_synth.py [data_dir]
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
from score import score, _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

THRESHOLD = 0.5
REL_BELOW = 0.05          # fee experiment: admit A down to 5% BELOW B, asymmetric


def _log(m=""):
    print(m, flush=True)


def _rule(c="-", n=104):
    _log(c * n)


# ----------------------------------------------------------------------------------
# Asymmetric amount block (fee experiment only)
# ----------------------------------------------------------------------------------
def _topk_asym(a, b, xa, xb, rel_below=REL_BELOW, top_k=C.TOP_K, tol_cents=C.TOL_CENTS):
    """Same as complete._topk but the amount block also admits candidates whose
    magnitude sits up to `rel_below` BELOW B's, same sign. Asymmetric on purpose: a fee
    reduces the settled amount, it never inflates it."""
    ak = (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy()
    bk = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    ad, bd = a["date"].to_numpy(), b["date"].to_numpy()
    ac, bc = a["cents"].to_numpy(), b["cents"].to_numpy()
    w = np.timedelta64(C.WINDOW, "D")

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
                tgt = bc[row]
                exact = np.abs(ca - tgt) <= tol_cents
                same_sign = np.sign(ca) == np.sign(tgt)
                lo, hi = abs(tgt) * (1.0 - rel_below), abs(tgt)
                fee_band = same_sign & (np.abs(ca) >= lo) & (np.abs(ca) <= hi)
                surv = np.flatnonzero(exact | fee_band)
                if surv.size == 0:
                    continue
                s = S[i, surv]
                k = min(top_k, s.size)
                part = np.argpartition(-s, k - 1)[:k]
                order = part[np.argsort(-s[part])]
                out_idx[row] = cand[surv[order]]
                out_scr[row] = s[order]
    return out_idx, out_scr


# ----------------------------------------------------------------------------------
# Classifier — BenchRec train only
# ----------------------------------------------------------------------------------
def train_classifier(data_dir):
    _log("Fitting completion classifier on BenchRec_cash_v1.0_train.csv (train only).")
    t0 = time.perf_counter()
    a, b = R._load_sides(os.path.join(data_dir, "BenchRec_cash_v1.0_train.csv"))
    lab = C._labels_from_column(b)
    _, xa, xb = C._build_index(a, b)
    idx, scr = C._topk(a, b, xa, xb)
    X, y, _, _ = C._features(a, b, idx, scr, lab, True)
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    _log(f"  {len(y):,} candidate decisions, {int(y.sum()):,} positive "
         f"({y.mean() * 100:.2f}%), fitted in {time.perf_counter() - t0:.1f}s")
    _log("  >>> This model is now FROZEN. It is not refitted on synthetic data.")
    return clf


# ----------------------------------------------------------------------------------
# Pipeline over one batch
# ----------------------------------------------------------------------------------
def run_batch(tx_path, sol_path, clf, ranker=C._topk, threshold=THRESHOLD, label=""):
    t0 = time.perf_counter()
    a, b = R._load_sides(tx_path)
    lab = R._labels(sol_path)
    _, xa, xb = C._build_index(a, b)
    idx, scr = ranker(a, b, xa, xb)

    X, _, owner, cand_key = C._features(a, b, idx, scr, lab, True)
    proba = clf.predict_proba(X)[:, 1] if len(X) else np.empty(0)

    alloc = a["A_allocation"].to_numpy()
    b_ids = b["B_id"].to_numpy()
    p_off = C._assemble(b_ids, idx, alloc, proba, owner, cand_key, threshold=2.0)
    p_on = C._assemble(b_ids, idx, alloc, proba, owner, cand_key, threshold=threshold)
    wall = time.perf_counter() - t0

    n_rows = len(a) + len(b)
    return {"a": a, "b": b, "labels": lab, "idx": idx, "alloc": alloc,
            "p_off": p_off, "p_on": p_on, "wall": wall, "n_rows": n_rows,
            "n_b": len(b), "n_a": len(a), "label": label,
            "n_accepted": int((proba >= threshold).sum()),
            "n_decisions": len(proba)}


# ----------------------------------------------------------------------------------
# Per-class reporting
# ----------------------------------------------------------------------------------
def _class_map(manifest):
    cls, corr = {}, {}
    for g in manifest["groups"]:
        for bid in g["b_ids"]:
            cls[str(bid)] = g["class"]
            corr[str(bid)] = g["corruptions"]
    return cls, corr


def per_class(p_off, p_on, labels, manifest):
    cls, corr = _class_map(manifest)
    off = dict(zip(p_off["B_id"].astype(str), p_off["targetAllocation"]))
    on = dict(zip(p_on["B_id"].astype(str), p_on["targetAllocation"]))

    recs = []
    for bid, gold in labels.items():
        bid = str(bid)
        po, pn = _parse_alloc(off.get(bid, "")), _parse_alloc(on.get(bid, ""))
        # "predicted" follows score.score exactly: a row counts as predicted unless it
        # ABSTAINED, and abstention requires an empty prediction against a NON-empty
        # label. A correct blank on an unmatchable row is therefore a prediction, not an
        # abstention. Using "non-empty prediction" here instead would silently disagree
        # with the scorer on every blank-label row.
        recs.append({"cls": cls.get(bid, "?"), "corr": corr.get(bid, []),
                     "c_off": po == gold, "p_off": not (len(po) == 0 and len(gold) > 0),
                     "c_on": pn == gold, "p_on": not (len(pn) == 0 and len(gold) > 0)})
    d = pd.DataFrame(recs)

    def block(sel, name):
        if not len(sel):
            return None
        po, pn = sel["p_off"].sum(), sel["p_on"].sum()
        return {"class": name, "rows": len(sel),
                "match_off_%": round(sel["c_off"].mean() * 100, 3),
                "match_on_%": round(sel["c_on"].mean() * 100, 3),
                "prec_off_%": round(sel[sel["p_off"]]["c_off"].mean() * 100, 3) if po else None,
                "prec_on_%": round(sel[sel["p_on"]]["c_on"].mean() * 100, 3) if pn else None}

    rows = [block(g, c) for c, g in d.groupby("cls")]
    for nm in ("fee_deduction", "timing_offset", "duplicate_reference"):
        rows.append(block(d[d["corr"].map(lambda x: nm in x)], f"[corruption] {nm}"))
    rows.append(block(d, "ALL"))
    return pd.DataFrame([r for r in rows if r])


def repeat_recovery(p_off, p_on, labels, manifest):
    cls, _ = _class_map(manifest)
    off = dict(zip(p_off["B_id"].astype(str), p_off["targetAllocation"]))
    on = dict(zip(p_on["B_id"].astype(str), p_on["targetAllocation"]))

    full = partial_all_correct = added_wrong = none_added = 0
    n = 0
    for bid, gold in labels.items():
        bid = str(bid)
        if cls.get(bid) != "repeat":
            continue
        n += 1
        po, pn = _parse_alloc(off.get(bid, "")), _parse_alloc(on.get(bid, ""))
        added = pn - po
        if pn == gold:
            full += 1
        elif not added:
            none_added += 1
        elif pn <= gold:
            partial_all_correct += 1      # everything added was right, but incomplete
        else:
            added_wrong += 1              # at least one added key is not in the label
    return {"repeat_groups": n, "full_set_recovered": full,
            "added_some_all_correct_but_incomplete": partial_all_correct,
            "added_at_least_one_wrong_key": added_wrong,
            "no_keys_added": none_added}


def _overall(p, sol_path):
    r = score(p, sol_path)
    return r["match_rate"] * 100, r["match_precision"] * 100


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    _rule("=")
    _log("FULL PIPELINE OVER SYNTHETIC BATCHES — retriever + completion @ 0.5")
    _rule("=")
    _log()
    _log("  *** OUT-OF-DOMAIN RUN ***")
    _log("  The completion classifier is trained on BenchRec_cash_v1.0_train.csv and is")
    _log("  applied here WITHOUT retraining, refitting or recalibration on synthetic data.")
    _log("  Every synthetic number below is an out-of-domain transfer result. Testing that")
    _log("  transfer is part of the point of this run.")
    _log()

    clf = train_classifier(data_dir)

    batches = [
        ("synth_small", "50-group batch"),
        ("synth", "50,000-group batch"),
    ]

    stored = {}
    for prefix, name in batches:
        tx = os.path.join(data_dir, f"{prefix}_transactions.csv")
        sol = os.path.join(data_dir, f"{prefix}_solution.csv")
        man = json.load(open(os.path.join(data_dir, f"{prefix}_manifest.json"),
                             encoding="utf-8"))

        _log()
        _rule("=")
        _log(f"BATCH: {name}   ({prefix}_*)")
        _rule("=")

        res = run_batch(tx, sol, clf, ranker=C._topk, label=name)
        stored[prefix] = (res, man, sol)

        mo, po = _overall(res["p_off"], sol)
        mn, pn = _overall(res["p_on"], sol)
        _log()
        _log(f"  overall  completion OFF  match {mo:7.3f}%  precision {po:7.3f}%")
        _log(f"           completion ON   match {mn:7.3f}%  precision {pn:7.3f}%")
        _log()
        _log("  PER-CLASS (classes from the manifest; corruption cuts overlap the")
        _log("  structure classes and each other, so they are listed separately):")
        _log()
        tab = per_class(res["p_off"], res["p_on"], res["labels"], man)
        _log(tab.to_string(index=False))

        # The ALL row must reproduce the existing scorer exactly.
        allrow = tab[tab["class"] == "ALL"].iloc[0]
        # table values are rounded to 3dp, so compare at that resolution
        ok = (abs(allrow["match_off_%"] - mo) < 1e-3 and
              abs(allrow["prec_off_%"] - po) < 1e-3 and
              abs(allrow["match_on_%"] - mn) < 1e-3 and
              abs(allrow["prec_on_%"] - pn) < 1e-3)
        _log()
        _log(f"  ALL row reconciles with score.score(): {ok}")

        _log()
        _log("  THROUGHPUT")
        _log(f"      groups                 {man['n_groups']:>12,}")
        _log(f"      total rows processed   {res['n_rows']:>12,}   "
             f"(A {res['n_a']:,}  B {res['n_b']:,})")
        _log(f"      wall clock             {res['wall']:>12.2f} s   "
             f"(rank + classify + assemble; excludes one-off classifier fit)")
        _log(f"      rows/sec               {res['n_rows'] / res['wall']:>12,.1f}")
        _log(f"      B records/sec          {res['n_b'] / res['wall']:>12,.1f}")
        if man["n_groups"] < 100:
            _log(f"      -> {res['n_rows']:,} rows / {res['n_b']:,} B records / "
                 f"{man['n_groups']} groups: satisfies the stated 50+ record minimum.")

        rr = repeat_recovery(res["p_off"], res["p_on"], res["labels"], man)
        _log()
        _log("  REPEAT CLASS — what completion actually recovered")
        _log(f"      repeat groups                              {rr['repeat_groups']:>8,}")
        _log(f"      FULL key set recovered (exact match)       "
             f"{rr['full_set_recovered']:>8,}")
        _log(f"      added some, all correct, still incomplete  "
             f"{rr['added_some_all_correct_but_incomplete']:>8,}")
        _log(f"      added at least one WRONG key               "
             f"{rr['added_at_least_one_wrong_key']:>8,}")
        _log(f"      added nothing (left as single key)         {rr['no_keys_added']:>8,}")

    # ------------------------------------------------------------------
    # Fee experiment
    # ------------------------------------------------------------------
    _log()
    _rule("=")
    _log(f"FEE EXPERIMENT — amount block widened to admit A down to {REL_BELOW * 100:.0f}% "
         f"BELOW B (asymmetric)")
    _rule("=")
    _log()
    _log("  Baseline block: |A - B| <= 0.01, symmetric.")
    _log(f"  Widened block:  that, OR  {1 - REL_BELOW:.2f}*|B| <= |A| <= |B|  with matching sign.")
    _log("  Asymmetric because a fee reduces the settled amount and never inflates it.")
    _log("  This is ONE setting, reported as-is. The width was not tuned.")
    _log("  The classifier is still the frozen BenchRec-train model — now applied over a")
    _log("  candidate set built differently from the one it was fitted on, which is a")
    _log("  second domain shift on top of the synthetic one.")

    for prefix, name in batches:
        res_base, man, sol = stored[prefix]
        tx = os.path.join(data_dir, f"{prefix}_transactions.csv")

        _log()
        _rule()
        _log(f"  {name}")
        _rule()
        res_w = run_batch(tx, sol, clf, ranker=_topk_asym, label=name + " widened")

        base = per_class(res_base["p_off"], res_base["p_on"], res_base["labels"], man)
        wide = per_class(res_w["p_off"], res_w["p_on"], res_w["labels"], man)
        m = base.merge(wide, on="class", suffixes=("_base", "_wide"))
        m["d_match_on"] = (m["match_on_%_wide"] - m["match_on_%_base"]).round(3)
        m["d_prec_on"] = (m["prec_on_%_wide"] - m["prec_on_%_base"]).round(3)
        _log()
        _log(m[["class", "rows_base", "match_on_%_base", "match_on_%_wide", "d_match_on",
                "prec_on_%_base", "prec_on_%_wide", "d_prec_on"]].to_string(index=False))

        mo, po = _overall(res_base["p_on"], sol)
        mw, pw = _overall(res_w["p_on"], sol)
        _log()
        _log(f"      overall match     {mo:7.3f}%  ->  {mw:7.3f}%   ({mw - mo:+.3f} pts)")
        _log(f"      overall precision {po:7.3f}%  ->  {pw:7.3f}%   ({pw - po:+.3f} pts)")
        _log(f"      wall clock        {res_base['wall']:.2f}s  ->  {res_w['wall']:.2f}s")


if __name__ == "__main__":
    _main()
