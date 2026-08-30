"""
exposure.py — value-weighted measurement over decisions already made.

Imports the existing modules unmodified. Reads the audit records controller.py wrote and
the solution files, and reports what the decisions are worth in money rather than in rows.
It tunes nothing, changes nothing, and re-decides nothing: every decision and every
answer is taken verbatim from the audit.

EXPOSURE DEFINITION (stated in the output too, because the number is meaningless
without it):

  * A wrongly auto-closed row contributes its FULL absolute B amount. The system posted
    an answer that was not correct and nobody is going to look at it, so the whole
    transaction value is exposed, not the difference between two candidates.

  * An escalated row contributes its FULL absolute B amount as value at risk pending
    review. This includes rows whose correct answer is blank: until a reviewer confirms
    that nothing should have matched, the amount is unresolved. Escalated exposure is
    therefore workload, not loss — it is the value awaiting a human, whereas wrongly
    auto-closed exposure is value that has already gone out the door unchecked.

  * A correctly auto-closed row contributes nothing to exposure. It is counted only in
    the denominator of value-weighted precision.

Run:  python exposure.py [data_dir]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import controller as CTL           # unmodified — trigger and exception-class vocabulary
from score import _parse_alloc     # unmodified — the same parser the scorer uses

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BATCHES = [
    ("BenchRec eval", "controller_audit_eval.jsonl",
     "BenchRec_cash_v1.0_solution.csv", "exceptions_ranked_eval.csv"),
    ("synthetic 50,000-group", "controller_audit_synth.jsonl",
     "synth_solution.csv", "exceptions_ranked_synth.csv"),
]

DECILES = [1, 5, 10, 25, 50, 75, 100]
RANDOM_SEEDS = 20


def _log(m=""):
    print(m, flush=True)


def _rule(c="-", n=104):
    _log(c * n)


def _money(cents) -> str:
    """Cents -> a readable magnitude. Values here span cents to billions."""
    d = cents / 100.0
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(d) >= div:
            return f"{d / div:,.2f}{suf}"
    return f"{d:,.2f}"


# ----------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------
def load_batch(data_dir, audit_file, solution_file):
    sol = pd.read_csv(os.path.join(data_dir, solution_file), dtype=str,
                      keep_default_na=False)
    labels = {str(b): _parse_alloc(t)
              for b, t in zip(sol["B_id"], sol["targetAllocation"])}

    rows = []
    with open(os.path.join(data_dir, audit_file), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            bid = str(d["b_id"])
            answer = set(d.get("answer_keys") or [])
            gold = labels.get(bid, set())
            rows.append({
                "b_id": bid,
                "exposure_cents": abs(int(d["b_amount_cents"])),
                "b_amount_cents": int(d["b_amount_cents"]),
                "decision": d["decision"],
                "auto_closed": d["decision"] == "auto_close",
                "correct": answer == gold,
                "exception_class": d.get("exception_class") or "",
                "triggers": d.get("triggers") or [],
                "gold_blank": len(gold) == 0,
                "top1_score": d.get("top1_score"),
                "margin": d.get("margin"),
                "pool_size": d.get("pool_size"),
                "n_candidates": len(d.get("candidates") or []),
                "exact_amount_top1": d.get("exact_amount_top1"),
                "dup_ref": d.get("duplicate_reference_among_candidates"),
                "n_added_keys": len(d.get("added_keys") or []),
                "_candidates": d.get("candidates") or [],
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Headline: row-count vs value-weighted, side by side
# ----------------------------------------------------------------------------------
def headline(df, name):
    _log()
    _rule("=")
    _log(f"ROW-COUNT vs VALUE-WEIGHTED — {name}")
    _rule("=")
    _log()

    auto = df[df["auto_closed"]]
    esc = df[~df["auto_closed"]]
    ac_ok = auto[auto["correct"]]
    ac_bad = auto[~auto["correct"]]

    tot_v = df["exposure_cents"].sum()
    auto_v = auto["exposure_cents"].sum()
    ok_v = ac_ok["exposure_cents"].sum()
    bad_v = ac_bad["exposure_cents"].sum()
    esc_v = esc["exposure_cents"].sum()

    tab = pd.DataFrame([
        {"metric": "auto-close precision",
         "by_rows": f"{len(ac_ok) / len(auto) * 100:.3f}%" if len(auto) else "n/a",
         "by_value": f"{ok_v / auto_v * 100:.3f}%" if auto_v else "n/a"},
        {"metric": "auto-closed",
         "by_rows": f"{len(auto):,} ({len(auto) / len(df) * 100:.2f}%)",
         "by_value": f"{_money(auto_v)} ({auto_v / tot_v * 100:.2f}%)"},
        {"metric": "escalated",
         "by_rows": f"{len(esc):,} ({len(esc) / len(df) * 100:.2f}%)",
         "by_value": f"{_money(esc_v)} ({esc_v / tot_v * 100:.2f}%)"},
    ])
    _log(tab.to_string(index=False))
    _log()
    _log(f"  total absolute B value in batch                 {_money(tot_v):>18}")
    _log(f"  value auto-closed CORRECTLY                     {_money(ok_v):>18}"
         f"   ({ok_v / tot_v * 100:.3f}% of batch)")
    _log(f"  value auto-closed INCORRECTLY  (exposure)       {_money(bad_v):>18}"
         f"   ({bad_v / tot_v * 100:.3f}% of batch)")
    _log(f"  value in the exception queue   (at risk)        {_money(esc_v):>18}"
         f"   ({esc_v / tot_v * 100:.3f}% of batch)")
    _log()
    _log(f"  rows auto-closed incorrectly                    {len(ac_bad):>18,}")
    _log(f"  mean value of a wrongly auto-closed row         "
         f"{_money(ac_bad['exposure_cents'].mean()) if len(ac_bad) else '-':>18}")
    _log(f"  mean value of a correctly auto-closed row       "
         f"{_money(ac_ok['exposure_cents'].mean()) if len(ac_ok) else '-':>18}")
    _log(f"  mean value of an escalated row                  "
         f"{_money(esc['exposure_cents'].mean()) if len(esc) else '-':>18}")

    row_prec = len(ac_ok) / len(auto) * 100 if len(auto) else float("nan")
    val_prec = ok_v / auto_v * 100 if auto_v else float("nan")
    _log()
    delta = val_prec - row_prec
    if abs(delta) < 0.05:
        _log("  Value-weighted precision tracks row-count precision: errors are not "
             "concentrated in large or small transactions.")
    elif delta < 0:
        _log(f"  Value-weighted precision is {abs(delta):.3f} points WORSE than row-count "
             "precision.")
        _log("  Errors fall disproportionately on larger amounts — the rows it gets wrong "
             "are worth more")
        _log("  than the average row it closes.")
    else:
        _log(f"  Value-weighted precision is {delta:.3f} points BETTER than row-count "
             "precision.")
        _log("  Errors fall disproportionately on smaller amounts.")
    return {"tot": tot_v, "auto": auto_v, "ok": ok_v, "bad": bad_v, "esc": esc_v}


# ----------------------------------------------------------------------------------
# Queue value by class and by trigger
# ----------------------------------------------------------------------------------
def queue_breakdown(df, name):
    esc = df[~df["auto_closed"]]
    if not len(esc):
        return
    tot = esc["exposure_cents"].sum()

    _log()
    _rule("=")
    _log(f"EXCEPTION QUEUE VALUE — {name}")
    _rule("=")
    _log()
    _log("  'would_be_correct' is value the system would have got right had it closed")
    _log("  blind: coverage given up. 'would_be_wrong' is value the escalation caught.")
    _log()
    _log("  By exception class (mutually exclusive):")
    rows = []
    for cls in sorted(esc["exception_class"].unique()):
        g = esc[esc["exception_class"] == cls]
        v = g["exposure_cents"].sum()
        rows.append({
            "exception_class": cls or "(none)",
            "rows": len(g),
            "exposure": _money(v),
            "pct_of_queue_value": round(v / tot * 100, 3),
            "would_be_correct": _money(g[g["correct"]]["exposure_cents"].sum()),
            "would_be_wrong": _money(g[~g["correct"]]["exposure_cents"].sum()),
            "mean_row_value": _money(g["exposure_cents"].mean()),
        })
    _log(pd.DataFrame(rows).sort_values("pct_of_queue_value", ascending=False)
         .to_string(index=False))

    _log()
    _log("  By trigger (NON-exclusive — one row can trip several, so these sum to more")
    _log("  than the queue total):")
    rows = []
    for t in CTL.TRIGGERS:
        m = esc["triggers"].map(lambda x: t in x)
        g = esc[m]
        if not len(g):
            rows.append({"trigger": t, "rows": 0, "exposure": _money(0),
                         "pct_of_queue_value": 0.0, "would_be_correct": _money(0),
                         "would_be_wrong": _money(0)})
            continue
        v = g["exposure_cents"].sum()
        rows.append({
            "trigger": t, "rows": len(g), "exposure": _money(v),
            "pct_of_queue_value": round(v / tot * 100, 3),
            "would_be_correct": _money(g[g["correct"]]["exposure_cents"].sum()),
            "would_be_wrong": _money(g[~g["correct"]]["exposure_cents"].sum()),
        })
    _log(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------------
# Ranked queue + retired curve
# ----------------------------------------------------------------------------------
def ranked_queue(df, data_dir, out_file, name):
    esc = df[~df["auto_closed"]].sort_values("exposure_cents",
                                             ascending=False).reset_index(drop=True)
    if not len(esc):
        return None

    ev = []
    for _, r in esc.iterrows():
        ev.append(json.dumps({
            "top1_score": r["top1_score"], "margin": r["margin"],
            "pool_size": r["pool_size"], "n_candidates": r["n_candidates"],
            "exact_amount_top1": r["exact_amount_top1"],
            "duplicate_reference_among_candidates": r["dup_ref"],
            "n_keys_added_by_completion": r["n_added_keys"],
            "candidate_scores": [c.get("score") for c in r["_candidates"]],
            "candidate_amount_deltas_cents": [c.get("amount_delta_cents")
                                              for c in r["_candidates"]],
        }, separators=(",", ":")))

    out = pd.DataFrame({
        "rank": range(1, len(esc) + 1),
        "b_id": esc["b_id"],
        "exposure": (esc["exposure_cents"] / 100).round(2),
        "exposure_cents": esc["exposure_cents"],
        "exception_class": esc["exception_class"],
        "triggers": esc["triggers"].map(lambda x: "|".join(x)),
        "evidence": ev,
    })
    path = os.path.join(data_dir, out_file)
    out.to_csv(path, index=False)
    _log()
    _log(f"  ranked queue -> {out_file}  ({len(out):,} rows, "
         f"{_money(esc['exposure_cents'].sum())} total exposure)")
    return esc


def retired_curve(esc, name, seeds=RANDOM_SEEDS):
    _log()
    _rule("=")
    _log(f"EXPOSURE-RETIRED CURVE — {name}")
    _rule("=")
    _log()
    _log("  What fraction of total queue exposure is retired by reviewing the top N% of")
    _log("  the queue, under exposure-descending order vs a random order. Ranking only")
    _log(f"  earns its keep if it beats random (mean of {seeds} seeds).")
    _log()

    v = esc["exposure_cents"].to_numpy()
    total = v.sum()
    n = len(v)

    rng = np.random.default_rng(0)
    rand_curves = []
    for s in range(seeds):
        p = rng.permutation(n)
        rand_curves.append(np.cumsum(v[p]) / total)
    rand_mat = np.vstack(rand_curves)
    ranked_cum = np.cumsum(v) / total          # already sorted descending

    rows = []
    for pct in DECILES:
        k = max(1, int(round(n * pct / 100)))
        rows.append({
            "review_top_%": pct,
            "rows_reviewed": k,
            "exposure_retired_ranked_%": round(float(ranked_cum[k - 1]) * 100, 3),
            "exposure_retired_random_%": round(float(rand_mat[:, k - 1].mean()) * 100, 3),
            "random_sd": round(float(rand_mat[:, k - 1].std()) * 100, 3),
        })
    tab = pd.DataFrame(rows)
    tab["ranked_minus_random"] = (tab["exposure_retired_ranked_%"]
                                  - tab["exposure_retired_random_%"]).round(3)
    _log(tab.to_string(index=False))
    _log()
    r10 = tab[tab["review_top_%"] == 10].iloc[0]
    _log(f"  Reviewing the top 10% of the queue by exposure retires "
         f"{r10['exposure_retired_ranked_%']:.3f}% of queue value,")
    _log(f"  against {r10['exposure_retired_random_%']:.3f}% for random order — "
         f"a gain of {r10['ranked_minus_random']:.3f} points.")
    _log("  Random tracks the diagonal by construction; that is the point of showing it.")


# ----------------------------------------------------------------------------------
# The magnitude diagnostic
# ----------------------------------------------------------------------------------
def magnitude_buckets(df, name):
    _log()
    _rule("=")
    _log(f"RELIABILITY BY AMOUNT MAGNITUDE — {name}")
    _rule("=")
    _log()
    _log("  Buckets are powers of ten in absolute value. This is the diagnostic that says")
    _log("  whether the system is more or less trustworthy as the money gets bigger.")
    _log()

    d = df.copy()
    dollars = d["exposure_cents"] / 100.0
    # log10 bucket; guard zero/sub-cent
    with np.errstate(divide="ignore"):
        e = np.where(dollars > 0, np.floor(np.log10(dollars.replace(0, np.nan))), -3)
    d["exp10"] = np.nan_to_num(e, nan=-3).astype(int)

    def label(k):
        if k < 0:
            return f"< $1  (1e{k})" if k > -3 else "< $1"
        lo = 10 ** k
        return f"$1e{k} – 1e{k + 1}" if k >= 3 else f"${lo:,} – ${10 ** (k + 1):,}"

    rows = []
    for k in sorted(d["exp10"].unique()):
        g = d[d["exp10"] == k]
        auto = g[g["auto_closed"]]
        rows.append({
            "bucket": label(int(k)),
            "rows": len(g),
            "pct_of_rows": round(len(g) / len(d) * 100, 3),
            "value": _money(g["exposure_cents"].sum()),
            "escalation_rate_%": round((~g["auto_closed"]).mean() * 100, 3),
            "auto_close_precision_%": (round(auto["correct"].mean() * 100, 3)
                                       if len(auto) else None),
            "auto_closed_rows": len(auto),
        })
    tab = pd.DataFrame(rows)
    _log(tab.to_string(index=False))

    # Read the trend off the buckets that carry enough rows to mean anything.
    solid = tab[(tab["auto_closed_rows"] >= 30) & tab["auto_close_precision_%"].notna()]
    _log()
    if len(solid) >= 3:
        x = np.arange(len(solid))
        y = solid["auto_close_precision_%"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        lo, hi = solid.iloc[0], solid.iloc[-1]
        _log(f"  Across buckets with >=30 auto-closed rows ({len(solid)} buckets):")
        _log(f"      smallest: {lo['bucket']:>22}  precision "
             f"{lo['auto_close_precision_%']:.3f}%  escalation {lo['escalation_rate_%']:.3f}%")
        _log(f"      largest:  {hi['bucket']:>22}  precision "
             f"{hi['auto_close_precision_%']:.3f}%  escalation {hi['escalation_rate_%']:.3f}%")
        _log(f"      trend in auto-close precision per bucket: {slope:+.3f} points")
        _log()
        if slope < -0.5:
            _log("  >>> The system is LESS reliable as amounts get larger. Auto-close")
            _log("  >>> precision falls with magnitude, which is the wrong direction: the")
            _log("  >>> rows it is most confident about are the ones worth most.")
        elif slope > 0.5:
            _log("  >>> The system is MORE reliable as amounts get larger.")
        else:
            _log("  >>> Auto-close precision is broadly flat across magnitude. The system")
            _log("  >>> is neither better nor worse on large amounts.")
    else:
        _log("  Too few buckets carry >=30 auto-closed rows to read a trend.")
    return tab


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    _rule("=")
    _log("EXPOSURE — value-weighted measurement over decisions already made")
    _rule("=")
    _log()
    _log("  Nothing here is tuned. Every decision, answer and trigger is read verbatim")
    _log("  from the controller's audit records; this only re-weights them by money.")
    _log()
    _log("  EXPOSURE DEFINITION")
    _log("    wrongly auto-closed : the FULL absolute B amount. A wrong answer was posted")
    _log("                          and nobody will look at it, so the whole transaction")
    _log("                          value is exposed — not a difference between candidates.")
    _log("    escalated           : the FULL absolute B amount, at risk pending review.")
    _log("                          This INCLUDES rows whose correct answer is blank: until")
    _log("                          a reviewer confirms nothing should have matched, the")
    _log("                          amount is unresolved.")
    _log("    correctly auto-closed: contributes nothing to exposure; counted only in the")
    _log("                          denominator of value-weighted precision.")
    _log()
    _log("    Escalated exposure is WORKLOAD (value awaiting a human). Wrongly auto-closed")
    _log("    exposure is value already out the door unchecked. Do not add them together.")

    for name, audit, sol, out in BATCHES:
        ap = os.path.join(data_dir, audit)
        if not os.path.exists(ap):
            _log(f"\n  {audit} not found — skipping {name}.")
            continue
        df = load_batch(data_dir, audit, sol)
        headline(df, name)
        queue_breakdown(df, name)
        esc = ranked_queue(df, data_dir, out, name)
        if esc is not None:
            retired_curve(esc, name)
        magnitude_buckets(df, name)


if __name__ == "__main__":
    _main()
