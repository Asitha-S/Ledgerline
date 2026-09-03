"""
datewindow.py — how much does the retriever depend on the date window?

The +/-7 day window in retrieve.py was chosen for BenchRec and never tested. Real
settlement timing is not symmetric and not 7 days: refunds land weeks after the payment,
bank holidays compound delays, international settlement runs T+7 rather than T+2. So the
question is where retrieval degrades as the window moves, and whether +/-7 sits on a
plateau or on an edge.

MEASUREMENT ONLY. retrieve.py is imported unmodified and its shipped configuration is
not changed: same loader, same TF-IDF feature space, same blocking key, same 0.01
reference amount tolerance, same cosine-on-full-text ranking, same top-5. The only thing
that varies between runs is the date interval. Nothing is fitted, tuned or re-decided.

WHAT THE WINDOW IS. A candidate ledger row survives blocking when it shares the
statement row's currency and account, its value date lies in the interval, and its
amount is within 0.01 of the statement amount. The interval is expressed here as
[lo, hi] days relative to the statement date, which makes the asymmetric variants
sayable: settlement delay runs in one direction, so "ledger before statement" is
[-w, 0] and "statement before ledger" is [0, +w]. Symmetric +/-w is [-w, +w].

SCOPE, stated because it bounds the conclusion. This sweeps RETRIEVAL: pool size, the
recall ceiling blocking imposes, top-1 and top-5, and the match rate and precision of
posting the retrieved answer blind. It does not re-run set completion or the decision
controller for each window — those would have to be refitted per window, which is
tuning, and refitting eleven classifiers to measure a blocking parameter would answer a
different question than the one asked. The retrieval ceiling bounds everything
downstream of it, so where the ceiling moves, the pipeline moves with it.

VALIDATION. At +/-7 this file must reproduce retrieve.log's shipped row exactly. If it
does not, the sweep is measuring something other than the shipped retriever and says so
rather than reporting numbers.

Run:  python datewindow.py [data_dir]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import retrieve as RET               # unmodified
from score import score, _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INF = 10 ** 6            # days; wider than any span in either dataset

# (label, lo, hi) in days relative to the statement date
SYMMETRIC = [("0", 0, 0), ("+/-1", -1, 1), ("+/-2", -2, 2), ("+/-3", -3, 3),
             ("+/-5", -5, 5), ("+/-7", -7, 7), ("+/-14", -14, 14),
             ("+/-30", -30, 30), ("+/-60", -60, 60), ("unbounded", -INF, INF)]
ASYMMETRIC = [("ledger-before only, 7", -7, 0), ("statement-before only, 7", 0, 7),
              ("ledger-before only, 30", -30, 0), ("statement-before only, 30", 0, 30)]

BATCHES = [
    ("BenchRec eval", "BenchRec_cash_v1.0_eval.csv", "BenchRec_cash_v1.0_solution.csv"),
    ("synthetic 50,000-group", "synth_transactions.csv", "synth_solution.csv"),
]

_log, _rule = RET._log, RET._rule


def _pct(n, d):
    return float("nan") if not d else n / d * 100.0


# ----------------------------------------------------------------------------------
# One window
# ----------------------------------------------------------------------------------
def sweep_one(P, lo, hi):
    """Block, rank and score at one date interval. Mirrors retrieve.run_all's shipped
    path exactly; only the interval differs."""
    a_keys, b_keys = P["a_keys"], P["b_keys"]
    a_alloc, a_date, b_date = P["a_alloc"], P["a_date"], P["b_date"]
    a_amt, b_amt = P["a_amt"], P["b_amt"]
    x_a, x_b = P["x_a"], P["x_b"]
    gold_rows, single = P["gold_rows"], P["single"]
    n_b = len(b_keys)

    w_lo = np.timedelta64(int(lo), "D")
    w_hi = np.timedelta64(int(hi), "D")

    alloc = np.array([""] * n_b, dtype=object)
    topk = [[] for _ in range(n_b)]
    pool_date = np.zeros(n_b, dtype=np.int64)
    pool_amt = np.zeros(n_b, dtype=np.int64)
    survived = np.zeros(n_b, dtype=bool)

    t0 = time.perf_counter()
    for _, b_idx in P["groups"].items():
        b_idx = np.asarray(b_idx)
        a_idx = np.where(a_keys == b_keys[b_idx[0]])[0]
        if len(a_idx) == 0:
            continue
        a_idx = a_idx[np.argsort(a_date[a_idx], kind="stable")]
        ds = a_date[a_idx]
        qd = b_date[b_idx[0]]
        cand = a_idx[np.searchsorted(ds, qd + w_lo, "left"):
                     np.searchsorted(ds, qd + w_hi, "right")]
        pool_date[b_idx] = len(cand)
        if len(cand) == 0:
            continue

        xa_c = x_a[cand]
        cand_amt = a_amt[cand]
        for start in range(0, len(b_idx), RET.QUERY_CHUNK):
            chunk = b_idx[start:start + RET.QUERY_CHUNK]
            s = (x_b[chunk] @ xa_c.T).toarray()
            absdiff = np.abs(cand_amt[None, :] - b_amt[chunk][:, None])
            keep = absdiff <= 1              # REFERENCE_TOL, in integer cents
            for i, row in enumerate(chunk):
                surv = np.flatnonzero(keep[i])
                pool_amt[row] = surv.size
                if surv.size == 0:
                    continue
                cs = cand[surv]
                # did any ledger row carrying the gold key survive blocking?
                g = gold_rows.get(row)
                if g is not None and len(np.intersect1d(cs, g, assume_unique=False)):
                    survived[row] = True
                sc = s[i, surv]
                k = min(RET.TOP_K, sc.size)
                part = np.argpartition(-sc, k - 1)[:k]
                order = part[np.argsort(-sc[part])]
                alloc[row] = a_alloc[cs[order[0]]]
                topk[row] = list(a_alloc[cs[order]])
    secs = time.perf_counter() - t0

    preds = pd.DataFrame({"B_id": P["b_ids"], "targetAllocation": alloc})
    res = score(preds[["B_id", "targetAllocation"]], P["solution_path"])
    sk = res["by_label_type"]["single_key"]

    hit1 = hit5 = 0
    for r in single:
        key = P["gold_key"][r]
        t = topk[r]
        if t and t[0] == key:
            hit1 += 1
        if key in t:
            hit5 += 1

    return {
        "pool_date_mean": float(np.mean(pool_date)),
        "pool_date_median": float(np.median(pool_date)),
        "pool_amt_mean": float(np.mean(pool_amt)),
        "pool_amt_median": float(np.median(pool_amt)),
        "ceiling_pct": _pct(int(survived[single].sum()), len(single)),
        "top1_pct": _pct(hit1, len(single)),
        "top5_pct": _pct(hit5, len(single)),
        "single_match_pct": sk["match_rate"] * 100,
        "single_prec_pct": None if sk["match_precision"] is None
                           else sk["match_precision"] * 100,
        "overall_match_pct": res["match_rate"] * 100,
        "abstain_pct": res["abstention_rate"] * 100,
        "secs": secs,
    }


# ----------------------------------------------------------------------------------
def prepare(data_dir, tx_file, solution_file):
    a, b = RET._load_sides(os.path.join(data_dir, tx_file))
    sol_path = os.path.join(data_dir, solution_file)
    labels = RET._labels(sol_path)

    vec = TfidfVectorizer(analyzer="char", ngram_range=RET.NGRAM_RANGE, min_df=2)
    vec.fit(pd.concat([a["text"], b["text_full"]], ignore_index=True))
    x_a = vec.transform(a["text"])
    x_b = vec.transform(b["text_full"])

    a_alloc = a["A_allocation"].to_numpy()
    b_ids = b["B_id"].to_numpy()
    # ledger rows per allocation key, so "did the true match survive" is a set test
    by_key = {}
    for i, k in enumerate(a_alloc):
        by_key.setdefault(k, []).append(i)
    by_key = {k: np.array(v) for k, v in by_key.items()}

    single, gold_rows, gold_key = [], {}, {}
    for r, bid in enumerate(b_ids):
        g = labels.get(str(bid), set())
        if len(g) != 1:
            continue
        k = next(iter(g))
        if k not in by_key:
            continue
        single.append(r)
        gold_rows[r] = by_key[k]
        gold_key[r] = k

    b_keys = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    gk = pd.Series(b_keys) + "||" + b["B_valueDate"].astype(str)

    return {
        "a_keys": (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy(),
        "b_keys": b_keys, "a_alloc": a_alloc, "b_ids": b_ids,
        "a_date": a["date"].to_numpy(), "b_date": b["date"].to_numpy(),
        "a_amt": a["cents"].to_numpy(), "b_amt": b["cents"].to_numpy(),
        "x_a": x_a, "x_b": x_b, "groups": gk.groupby(gk).indices,
        "gold_rows": gold_rows, "gold_key": gold_key,
        "single": np.array(single), "solution_path": sol_path,
        "n_a": len(a), "n_b": len(b),
    }


def timing_truth(data_dir):
    """What the synthetic generator actually injected, so the sweep can be checked
    against ground truth rather than only against itself."""
    p = os.path.join(data_dir, "synth_manifest.json")
    if not os.path.exists(p):
        return None
    import json
    m = json.load(open(p, encoding="utf-8"))
    cfg = m.get("config", {})
    out = {k: v for k, v in cfg.items() if "timing" in k.lower() or "date" in k.lower()
           or "day" in k.lower() or "offset" in k.lower()}
    classes = m.get("class_counts") or m.get("classes") or {}
    if isinstance(classes, dict):
        for k, v in classes.items():
            if "timing" in str(k).lower():
                out[f"groups::{k}"] = v
    return out or None


def observed_offsets(P):
    """The actual signed gap, in days, between a statement row and the ledger rows
    carrying its gold key. Ground truth about how much timing variation exists."""
    gaps = []
    for r in P["single"]:
        g = P["gold_rows"][r]
        d = (P["a_date"][g] - P["b_date"][r]) / np.timedelta64(1, "D")
        d = d[np.isfinite(d)]
        if d.size:
            gaps.append(d[np.argmin(np.abs(d))])
    return np.array(gaps)


# ----------------------------------------------------------------------------------
def report(name, P, data_dir):
    _log()
    _rule("=")
    _log(f"DATE WINDOW SWEEP — {name}")
    _rule("=")
    _log()
    _log(f"  {P['n_a']:,} ledger rows, {P['n_b']:,} statement rows, "
         f"{len(P['single']):,} single-key labels whose key exists on the ledger side")

    gaps = observed_offsets(P)
    if gaps.size:
        q = np.percentile(np.abs(gaps), [50, 90, 99, 100])
        _log()
        _log("  GROUND TRUTH — signed gap to the nearest ledger row carrying the gold key")
        _log(f"    within 0 days   {_pct(int((np.abs(gaps) <= 0).sum()), gaps.size):.3f}%")
        for d in (1, 2, 3, 5, 7, 14, 30, 60):
            _log(f"    within +/-{d:<3d}     {_pct(int((np.abs(gaps) <= d).sum()), gaps.size):.3f}%")
        _log(f"    |gap| median {q[0]:.1f}, p90 {q[1]:.1f}, p99 {q[2]:.1f}, max {q[3]:.1f} days")
        _log(f"    ledger BEFORE statement {_pct(int((gaps < 0).sum()), gaps.size):.3f}%, "
             f"same day {_pct(int((gaps == 0).sum()), gaps.size):.3f}%, "
             f"AFTER {_pct(int((gaps > 0).sum()), gaps.size):.3f}%")
        _log("    This is the distribution the window has to cover. Everything below is")
        _log("    the retriever's response to it.")

    rows = []
    for label, lo, hi in SYMMETRIC + ASYMMETRIC:
        r = sweep_one(P, lo, hi)
        r["window"] = label
        r["shipped"] = (lo, hi) == (-7, 7)
        rows.append(r)
        _log(f"    {label:<26} ceiling {r['ceiling_pct']:7.3f}%  "
             f"top1 {r['top1_pct']:7.3f}%  match {r['single_match_pct']:7.3f}%  "
             f"({r['secs']:.1f}s)")
    return pd.DataFrame(rows)


def table(df, title):
    _log()
    _rule()
    _log(title)
    _rule()
    _log()
    hdr = (f"  {'window':<26}{'pool(date)':>11}{'pool(+amt)':>11}{'ceiling%':>10}"
           f"{'top1%':>9}{'top5%':>9}{'match%':>9}{'prec%':>9}{'abstain%':>10}{'secs':>7}")
    _log(hdr)
    _log("  " + "-" * (len(hdr) - 2))
    for r in df.itertuples():
        mark = " *" if r.shipped else "  "
        _log(f"  {r.window + mark:<26}{r.pool_date_mean:>11.2f}{r.pool_amt_mean:>11.2f}"
             f"{r.ceiling_pct:>10.3f}{r.top1_pct:>9.3f}{r.top5_pct:>9.3f}"
             f"{r.single_match_pct:>9.3f}"
             f"{(r.single_prec_pct if r.single_prec_pct is not None else float('nan')):>9.3f}"
             f"{r.abstain_pct:>10.3f}{r.secs:>7.1f}")
    _log()
    _log("  * shipped. pool(date) is candidates after currency+account+date blocking;")
    _log("    pool(+amt) after the 0.01 amount tolerance, which is what gets ranked.")


def verdict(df, name):
    """Two questions, and they have different answers, so they are asked separately.

    The blocking CEILING can only rise as the window widens — more candidates can only
    add true matches. Ranking can only get harder — more candidates to be wrong about.
    Reporting one number for "sensitivity" hides that, so both curves are reported and
    the window is judged on where they cross."""
    sym = df[~df.window.str.contains("only")].reset_index(drop=True)
    ship = sym[sym.shipped].iloc[0]
    wide = sym.iloc[-1]
    _log()
    _rule()
    _log(f"WHERE IT MOVES — {name}")
    _rule()
    _log()

    _log(f"  {'window':<12}{'ceiling%':>10}{'d ceiling':>11}{'match%':>9}"
         f"{'d match':>9}{'prec%':>9}{'pool':>9}")
    _log("  " + "-" * 69)
    prev_c = prev_m = None
    for r in sym.itertuples():
        dc = "" if prev_c is None else f"{r.ceiling_pct - prev_c:+.3f}"
        dm = "" if prev_m is None else f"{r.single_match_pct - prev_m:+.3f}"
        _log(f"  {r.window + (' *' if r.shipped else ''):<12}{r.ceiling_pct:>10.3f}"
             f"{dc:>11}{r.single_match_pct:>9.3f}{dm:>9}"
             f"{(r.single_prec_pct or float('nan')):>9.3f}{r.pool_amt_mean:>9.2f}")
        prev_c, prev_m = r.ceiling_pct, r.single_match_pct
    _log()

    # ---- 1. the ceiling ---------------------------------------------------------
    best = sym["ceiling_pct"].max()
    knee = sym[sym["ceiling_pct"] >= best - 0.1].iloc[0]
    span = best - sym["ceiling_pct"].min()
    _log("  1. THE CEILING — what blocking makes reachable at all")
    _log(f"     0 days {sym.iloc[0]['ceiling_pct']:.3f}%  ->  unbounded {best:.3f}%   "
         f"total span {span:.3f} pts")
    _log(f"     narrowest window within 0.1 pts of the maximum: {knee['window']} "
         f"({knee['ceiling_pct']:.3f}%)")
    if span < 1.0:
        _log(f"     The window buys almost nothing: {span:.3f} points separate no window")
        _log("     at all from an unbounded one. Blocking on date is not what makes this")
        _log("     dataset retrievable.")
    else:
        _log(f"     The window matters: {span:.3f} points separate the narrowest setting")
        _log(f"     from the widest, and the curve saturates at {knee['window']}.")
    _log()

    # ---- 2. ranking -------------------------------------------------------------
    m0, mw = sym.iloc[0]["single_match_pct"], wide["single_match_pct"]
    p0, pw = sym.iloc[0]["single_prec_pct"], wide["single_prec_pct"]
    _log("  2. RANKING — what widening costs")
    _log(f"     match     0 days {m0:.3f}%  ->  unbounded {mw:.3f}%   ({mw - m0:+.3f} pts)")
    _log(f"     precision 0 days {p0:.3f}%  ->  unbounded {pw:.3f}%   ({pw - p0:+.3f} pts)")
    _log(f"     pool      0 days {sym.iloc[0]['pool_amt_mean']:.2f}  ->  "
         f"unbounded {wide['pool_amt_mean']:.2f} candidates per query")
    if mw < m0:
        _log("     Widening the window makes retrieval WORSE, not better. The extra")
        _log("     candidates it admits are overwhelmingly wrong ones, and every one of")
        _log("     them is another chance for cosine to rank a decoy first.")
    _log()

    # ---- 3. plateau or cliff ----------------------------------------------------
    _log("  3. IS +/-7 ON A PLATEAU OR AN EDGE?")
    lo = sym[sym.window.isin(["+/-3", "+/-5"])]
    hi = sym[sym.window.isin(["+/-14", "+/-30"])]
    drop_narrow = ship["ceiling_pct"] - lo["ceiling_pct"].min()
    gain_wide = hi["ceiling_pct"].max() - ship["ceiling_pct"]
    _log(f"     ceiling at +/-7                        {ship['ceiling_pct']:.3f}%")
    _log(f"     lost by narrowing to +/-3              {drop_narrow:.3f} pts")
    _log(f"     gained by widening to +/-30            {gain_wide:.3f} pts")
    _log(f"     match rate given up by widening to +/-30  "
         f"{ship['single_match_pct'] - hi['single_match_pct'].min():.3f} pts")
    if drop_narrow < 0.25 and gain_wide < 0.25:
        _log("     PLATEAU. Neither halving nor quadrupling the window moves the ceiling")
        _log("     by a quarter of a point. +/-7 is not a tuned value and nothing here")
        _log("     depends on it being 7. A deployment with slower settlement can widen")
        _log("     it safely, at a small and measurable cost in ranking precision.")
    else:
        _log("     EDGE. The ceiling is still climbing steeply below +/-7 and flattens at")
        _log(f"     or just past it: +/-3 gives up {drop_narrow:.3f} points that +/-7 recovers,")
        _log(f"     while going out to +/-30 adds only {gain_wide:.3f}. +/-7 sits at the knee,")
        _log("     which means it is the right value for THIS timing distribution and")
        _log("     would be the wrong one for a slower-settling batch.")
    _log()

    # ---- 4. asymmetry -----------------------------------------------------------
    _log("  4. ASYMMETRIC — settlement delay runs in one direction")
    asym = df[df.window.str.contains("only")]
    for r in asym.itertuples():
        base = sym[sym.window == ("+/-7" if r.window.endswith(", 7") else "+/-30")].iloc[0]
        _log(f"     {r.window:<26} ceiling {r.ceiling_pct:7.3f}% "
             f"({r.ceiling_pct - base.ceiling_pct:+.3f})  "
             f"match {r.single_match_pct:7.3f}% ({r.single_match_pct - base.single_match_pct:+.3f})  "
             f"pool {r.pool_amt_mean:6.2f} ({r.pool_amt_mean - base.pool_amt_mean:+.2f})")
    lb = asym[asym.window == "ledger-before only, 7"].iloc[0]
    sb = asym[asym.window == "statement-before only, 7"].iloc[0]
    _log(f"     One-sided at 7 costs {ship['ceiling_pct'] - max(lb['ceiling_pct'], sb['ceiling_pct']):.3f} "
         "points of ceiling against the symmetric window while")
    _log("     roughly halving the pool. Which side matters more is a property of the")
    _log("     data's own timing, printed as ground truth at the top of this section.")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    _log()
    _rule("=")
    _log("datewindow.py — retrieval sensitivity to the date window")
    _log("measurement only: retrieve.py imported unmodified, nothing tuned or refitted")
    _rule("=")

    for name, tx, sol in BATCHES:
        if not os.path.exists(os.path.join(data_dir, tx)):
            _log(f"\n[skip] {name}: {tx} not found")
            continue
        P = prepare(data_dir, tx, sol)
        df = report(name, P, data_dir)
        table(df, f"FULL SWEEP — {name}")

        if name.startswith("BenchRec"):
            ship = df[df.shipped].iloc[0]
            _log()
            _rule()
            _log("VALIDATION vs retrieve.log")
            _rule()
            _log()
            ok = True
            for lbl, got, want in (("single-key match", ship["single_match_pct"], 95.3688),
                                   ("single-key precision", ship["single_prec_pct"], 98.4138),
                                   ("top-1", ship["top1_pct"], 95.3688),
                                   ("overall match", ship["overall_match_pct"], 89.9900),
                                   ("abstention", ship["abstain_pct"], 3.8474)):
                good = abs(got - want) <= 0.001
                ok &= good
                _log(f"  {'ok  ' if good else 'FAIL'} {lbl:<22} "
                     f"this sweep {got:.4f}%   retrieve.log {want:.4f}%")
            if not ok:
                _log()
                _log("  The +/-7 row does not reproduce the shipped retriever, so this sweep is")
                _log("  measuring something else. Refusing to draw conclusions from it.")
                sys.exit(1)
            _log()
            _log("  The +/-7 row reproduces retrieve.log exactly, so the rest of the sweep")
            _log("  differs from the shipped retriever only in the date interval.")
        else:
            t = timing_truth(data_dir)
            if t:
                _log()
                _rule()
                _log("GENERATOR GROUND TRUTH — what timing variation was injected")
                _rule()
                _log()
                for k, v in sorted(t.items()):
                    _log(f"    {k:<44} {v}")

        verdict(df, name)

    _log()
    _rule("=")
    _log("done")
    _rule("=")


if __name__ == "__main__":
    main()
