"""
driftgate.py — can the drift blind spot be gated at decision time, without labels?

drift.py established that a wrong posting onto a key carrying the RIGHT amount produces
zero drift, so a balance check cannot see it. This script asks whether those rows are
identifiable from the audit alone, before any label is consulted, and prices the gate
that would follow.

It changes nothing. controller.py is not touched, no threshold is fitted, no decision is
re-made, no candidate is re-ranked. Every decision, answer and pool is read verbatim from
the audit controller.py already wrote, and the drift figures are drift.py's, imported
rather than re-derived, so the two reports cannot disagree.

------------------------------------------------------------------------------------
TWO WAYS TO COUNT "MORE THAN ONE EXACT-AMOUNT CANDIDATE", AND THEY DIFFER A LOT

    n_exact_keys   distinct allocation KEYS among the exact-amount candidates
    n_exact_cand   exact-amount candidate ROWS

An answer names keys, not rows, so five candidate rows carrying one allocation key are
one choice and not five. n_exact_keys is therefore the primary definition here and
n_exact_cand is reported beside it, because on BenchRec eval the two disagree sharply:
counting rows, 38.627% of auto-closed rows look ambiguous; counting keys, 7.386% do. A
gate built on the row count escalates five times as much work for one more catch, and
the difference is an artefact of how candidates are recorded rather than anything about
the decision.

SCOPE. Candidates are the top-5 the retriever surfaced (TOP_K = 5 in complete.py), not
the whole blocking pool, which is larger and recorded separately as pool_size. Both
counts are counts over the top-5, and this script says so rather than implying otherwise.

Every auto-closed row has at least one exact-amount candidate by construction:
controller.py escalates on t_fee = has & ~exact_top1, so a row whose best candidate
missed the amount never reaches auto-close. The question is whether more than one hit it.

------------------------------------------------------------------------------------
WHAT WOULD MAKE A GATE WORTH BUILDING

Two things, and both are reported. It has to CATCH the invisible errors at a rate well
above the base rate, and the correct auto-closes it also escalates have to be few enough
to be worth it. A gate with perfect recall that escalates a third of the batch is not a
control. Neither half is allowed to stand in for the other, and no threshold is searched
over: if the concentration is absent, that is reported as a failure rather than tuned
into a success.

Run:  python driftgate.py [data_dir]
"""

from __future__ import annotations

import collections
import os
import sys

import drift as D           # unmodified — same loader, same drift definition

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_log, _rule, _money, _pct = D._log, D._rule, D._money, D._pct

# The exchange rate this project already refused, for comparison. retrieve.log: the
# digit-run filter bought +1.0383 points of precision for 32.5615 points of match rate.
REFUSED_RATE = 32.5615 / 1.0383


def outcome(df):
    """The four outcomes drift.py can distinguish."""
    wrong = ~df["correct"]
    return {
        "correct": df["correct"],
        "wrong": wrong,
        "wrong_zero": wrong & df["exact"] & (df["drift_lo"] == 0),
        "wrong_nonzero": wrong & df["exact"] & (df["drift_lo"] != 0),
        "wrong_bounded": wrong & ~df["exact"],
    }


# ----------------------------------------------------------------------------------
# 1. Pool structure, crossed against drift outcome
# ----------------------------------------------------------------------------------
def section_structure(df, name):
    _log()
    _rule("=")
    _log(f"POOL STRUCTURE vs DRIFT OUTCOME — {name}")
    _rule("=")
    _log()
    _log(f"  auto-closed rows              {len(df):,}")
    _log(f"  candidate rows recorded       "
         f"{dict(sorted(collections.Counter(df['n_cands']).items()))}")
    _log(f"  exact-amount candidate ROWS   "
         f"{dict(sorted(collections.Counter(df['n_exact_cand']).items()))}")
    _log(f"  exact-amount candidate KEYS   "
         f"{dict(sorted(collections.Counter(df['n_exact_keys']).items()))}")
    _log(f"  duplicate_reference set       {int(df['dup_ref'].sum()):,} "
         f"({_pct(int(df['dup_ref'].sum()), len(df))})")
    _log()
    _log(f"  more than one exact ROW  {(df['n_exact_cand'] > 1).mean() * 100:.3f}%"
         f"      more than one exact KEY  {(df['n_exact_keys'] > 1).mean() * 100:.3f}%")
    ident = int((df["n_exact_cand"] == df["n_cands"]).sum())
    _log(f"  rows where every recorded candidate matched the amount exactly   "
         f"{ident:,} of {len(df):,}")
    if ident == len(df):
        _log("    On this batch the exact_amount flag adds nothing to the candidate count:")
        _log("    amount blocking is tight enough that nothing reaches the top-5 without")
        _log("    matching to the cent. Inside the pool, amount carries no discriminating")
        _log("    information at all — it has already been spent selecting the pool.")
    _log()

    o = outcome(df)
    hdr = (f"  {'exact keys':>11}{'dup_ref':>9}{'rows':>9}{'correct':>10}{'wrong':>8}"
           f"{'w drift=0':>11}{'w drift!=0':>12}{'w bounded':>11}")
    _log(hdr)
    _log("  " + "-" * (len(hdr) - 2))
    for (ne, dr), sub in df.groupby(["n_exact_keys", "dup_ref"], sort=True):
        so = outcome(sub)
        _log(f"  {ne:>11}{str(dr):>9}{len(sub):>9,}{int(so['correct'].sum()):>10,}"
             f"{int(so['wrong'].sum()):>8,}{int(so['wrong_zero'].sum()):>11,}"
             f"{int(so['wrong_nonzero'].sum()):>12,}{int(so['wrong_bounded'].sum()):>11,}")
    _log("  " + "-" * (len(hdr) - 2))
    _log(f"  {'all':>11}{'':>9}{len(df):>9,}{int(o['correct'].sum()):>10,}"
         f"{int(o['wrong'].sum()):>8,}{int(o['wrong_zero'].sum()):>11,}"
         f"{int(o['wrong_nonzero'].sum()):>12,}{int(o['wrong_bounded'].sum()):>11,}")
    _log()
    _log("  'w bounded' are wrong postings whose drift drift.py could only bound, because")
    _log("  a key in the symmetric difference is carried by ledger rows of several amounts.")
    _log("  Whether those are zero-drift is not determined by the data, so they are never")
    _log("  counted as zero and never counted as non-zero.")


# ----------------------------------------------------------------------------------
# 2. Concentration
# ----------------------------------------------------------------------------------
def section_concentration(df):
    _log()
    _rule()
    _log("DO THE INVISIBLE ERRORS CONCENTRATE ON AMBIGUOUS POOLS?")
    _rule()
    _log()
    o = outcome(df)
    z = df[o["wrong_zero"]]
    _log(f"  zero-drift wrong postings (invisible to a balance check)   {len(z):,}")
    if not len(z):
        _log("  none — nothing to concentrate")
        return
    _log()
    for col, lbl in (("n_exact_keys", "exact-amount KEYS"),
                     ("n_exact_cand", "exact-amount ROWS")):
        base = float((df[col] > 1).mean())
        hit = int((z[col] > 1).sum())
        _log(f"  by {lbl:<20} >1 on {hit:,} of {len(z):,} ({_pct(hit, len(z))})"
             f"   base {base * 100:.3f}%"
             f"   lift {((hit / len(z)) / base):.2f}x" if base else "")
    _log(f"  exactly one exact KEY (no pool signal at all)  "
         f"{int((z['n_exact_keys'] == 1).sum()):,} of {len(z):,}")
    _log()
    dz = int(z["dup_ref"].sum())
    _log(f"  dup_ref set on {dz:,} of {len(z):,} ({_pct(dz, len(z))})"
         f"   base {df['dup_ref'].mean() * 100:.3f}%")
    _log()
    _log(f"  label strata   {dict(collections.Counter(z['stratum']).most_common())}")
    _log(f"  gold key present in the recorded top-5   {int(z['gold_in_pool'].sum()):,} "
         f"of {len(z):,}")
    if int(z["gold_in_pool"].sum()) < len(z):
        _log("    For the rest the correct key was never a candidate, so no re-ranking")
        _log("    could have fixed them. Escalation is the only available remedy.")


# ----------------------------------------------------------------------------------
# 3. Pricing
# ----------------------------------------------------------------------------------
def section_pricing(df, n_all):
    _log()
    _rule()
    _log("PRICING THE GATES   (applied to decisions already made, nothing refitted)")
    _rule()
    _log()
    o = outcome(df)
    cov0 = len(df) / n_all * 100
    prec0 = int(o["correct"].sum()) / len(df) * 100
    _log(f"  current   coverage {cov0:.3f}%   precision {prec0:.3f}%   "
         f"({len(df):,} of {n_all:,} auto-closed, {int(o['correct'].sum()):,} correct)")
    _log()

    k = df["n_exact_keys"] > 1
    r = df["n_exact_cand"] > 1
    d = df["dup_ref"].astype(bool)
    gates = [("exact keys > 1", k), ("exact rows > 1", r), ("dup_ref", d),
             ("keys>1 or dup", k | d), ("keys>1 and dup", k & d)]

    hdr = (f"  {'gate':<16}{'escalates':>11}{'wrong':>7}{'d=0':>6}{'d!=0':>7}{'bnd':>6}"
           f"{'correct esc':>13}{'ok/wrong':>10}{'coverage':>11}{'precision':>11}{'pts/pt':>9}")
    _log(hdr)
    _log("  " + "-" * (len(hdr) - 2))
    rows = []
    for lbl, m in gates:
        cw = int((o["wrong"] & m).sum())
        cok = int((o["correct"] & m).sum())
        after = int((~m).sum())
        cov = after / n_all * 100
        prec = int((o["correct"] & ~m).sum()) / after * 100 if after else float("nan")
        dp = prec - prec0
        rate = (abs(cov - cov0) / dp) if dp > 0 else float("inf")
        rows.append({"gate": lbl, "mask": m, "cw": cw, "cok": cok, "cov": cov,
                     "prec": prec, "rate": rate,
                     "z": int((o["wrong_zero"] & m).sum())})
        _log(f"  {lbl:<16}{int(m.sum()):>11,}{cw:>7,}"
             f"{int((o['wrong_zero'] & m).sum()):>6,}"
             f"{int((o['wrong_nonzero'] & m).sum()):>7,}"
             f"{int((o['wrong_bounded'] & m).sum()):>6,}"
             f"{cok:>13,}{(cok / cw if cw else 0):>10.1f}"
             f"{cov:>10.3f}%{prec:>10.3f}%"
             f"{(f'{rate:.1f}' if rate != float('inf') else 'n/a'):>9}")
    _log()
    _log(f"  Baseline coverage {cov0:.3f}%, precision {prec0:.3f}%.")
    _log("  'ok/wrong'  correct auto-closes given up per wrong posting caught.")
    _log("  'pts/pt'    points of coverage surrendered per point of precision gained.")
    _log("  'bnd'       wrong postings of undetermined drift, caught incidentally; real")
    _log("              errors, but not evidence about the blind spot either way.")
    _log()
    _log(f"  For scale: this project refused the digit-run filter at "
         f"{REFUSED_RATE:.1f} points per point")
    _log("  (+1.0383 precision for 32.5615 match rate, retrieve.log; finding 3). That is a")
    _log("  stated yardstick, not a fitted one, and a reader may disagree with it.")
    return cov0, prec0, rows


# ----------------------------------------------------------------------------------
# 4. Residue
# ----------------------------------------------------------------------------------
def section_residue(df):
    _log()
    _rule()
    _log("THE RESIDUE — invisible errors no pool-structure gate could catch")
    _rule()
    _log()
    o = outcome(df)
    z = df[o["wrong_zero"]]
    if not len(z):
        _log("  none")
        return
    one = z[z["n_exact_keys"] == 1]
    _log(f"  invisible errors with exactly one exact-amount KEY   {len(one):,} of {len(z):,} "
         f"({_pct(len(one), len(z))})")
    _log(f"  ... and dup_ref unset as well                        "
         f"{int((one['n_exact_keys'] == 1).sum() and (~one['dup_ref'].astype(bool)).sum()):,}")
    _log()
    if not len(one):
        _log("  There is no residue: every invisible error sat in a pool that offered more")
        _log("  than one exact-amount key, so the signal is present on all of them.")
        _log("  Whether acting on it is affordable is the pricing table's question.")
        return
    _log(f"  {'B_id':<16}{'|B|':>15}{'keys':>6}{'rows':>6}{'pool':>9}{'top1':>9}"
         f"{'margin':>9}{'dup':>6}{'gold in top5':>14}  label")
    _log("  " + "-" * 106)
    for x in one.sort_values("abs_b", ascending=False).head(20).itertuples():
        _log(f"  {x.b_id:<16}{_money(x.abs_b):>15}{x.n_exact_keys:>6}{x.n_exact_cand:>6}"
             f"{x.pool_size:>9,}{x.top1_score:>9.4f}{x.margin:>9.4f}"
             f"{str(x.dup_ref):>6}{str(x.gold_in_pool):>14}  {x.stratum}")
    if len(one) > 20:
        _log(f"  ... and {len(one) - 20:,} more")
    _log()
    _log(f"  value mis-attributed by the residue   {_money(int(one['abs_b'].sum()))}")
    _log(f"  gold key was in the recorded top-5    {int(one['gold_in_pool'].sum()):,} of "
         f"{len(one):,}")
    _log()
    _log("  One exact-amount key, taken, and wrong, with the right key carrying the same")
    _log("  value from outside the candidate set. From inside the pool the row looks like")
    _log("  an unambiguous single hit. Pool structure cannot see that.")


# ----------------------------------------------------------------------------------
# 5. Verdict
# ----------------------------------------------------------------------------------
def section_verdict(df, n_all, cov0, prec0, rows, name):
    _log()
    _rule()
    _log("VERDICT")
    _rule()
    _log()
    o = outcome(df)
    z = df[o["wrong_zero"]]
    g = rows[0]                      # exact keys > 1, the primary gate
    if not len(z):
        _log("  No invisible errors on this batch.")
        return None
    hit = int((z["n_exact_keys"] > 1).sum())
    recall = hit / len(z)
    base = float((df["n_exact_keys"] > 1).mean())
    lift = recall / base if base else float("nan")

    _log("  1. Is the blindness predictable without labels?")
    if recall >= 0.9 and lift >= 2:
        _log(f"     YES. {hit:,} of {len(z):,} invisible errors ({recall * 100:.2f}%) sit in pools")
        _log(f"     offering more than one exact-amount key, against a {base * 100:.3f}% base")
        _log(f"     rate — {lift:.2f}x lift. The signal is there at decision time.")
    else:
        _log(f"     NO. {hit:,} of {len(z):,} ({recall * 100:.2f}%) against a {base * 100:.3f}% base")
        _log(f"     rate, {lift:.2f}x lift. They do NOT concentrate on ambiguous pools. The")
        _log("     gate does not work, and is reported as failing rather than tuned.")
    _log()
    _log("  2. What would it cost?")
    _log(f"     {int(g['mask'].sum()):,} rows escalated, {g['cok']:,} of them correct — "
         f"{g['cok'] / g['cw']:.1f} correct")
    _log(f"     auto-closes given up per wrong posting caught. Coverage "
         f"{cov0:.3f}% -> {g['cov']:.3f}%")
    _log(f"     ({g['cov'] - cov0:+.3f} pts), precision {prec0:.3f}% -> {g['prec']:.3f}% "
         f"({g['prec'] - prec0:+.3f} pts).")
    _log()
    if g["rate"] == float("inf"):
        _log("     REFUSE. It costs coverage and returns no precision.")
        verdict = "refuse"
    elif g["rate"] >= REFUSED_RATE * 0.8:
        _log(f"     REFUSE. At {g['rate']:.1f} points per point this is the trade the project")
        _log("     already turned down; taking it here would be inconsistent, not cautious.")
        verdict = "refuse"
    else:
        _log(f"     NOT REFUSED ON PRICE. At {g['rate']:.1f} points per point this is")
        _log(f"     materially cheaper than the {REFUSED_RATE:.1f} the project refused. It still")
        _log(f"     costs {abs(g['cov'] - cov0):.3f} points of coverage — {int(g['mask'].sum()):,} more rows into a queue")
        _log("     someone works before a deadline. That is a capacity question, and this")
        _log("     script does not answer it. What it can say is that the measurement does")
        _log("     not rule the gate out.")
        verdict = "open"
    _log()
    _log("  3. What the gate would not buy. Escalating these rows surfaces them for review;")
    _log("     it does not resolve them. Where the correct key was never in the top-5,")
    _log(f"     review is the only remedy anyway — {int((~z['gold_in_pool']).sum()):,} of "
         f"{len(z):,} invisible errors are in")
    _log("     that position.")
    return {"batch": name, "recall": recall, "lift": lift, "rate": g["rate"],
            "d_cov": g["cov"] - cov0, "d_pre": g["prec"] - prec0,
            "esc": int(g["mask"].sum()), "verdict": verdict}


# ----------------------------------------------------------------------------------
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    _log()
    _rule("=")
    _log("driftgate.py — can the drift blind spot be gated at decision time?")
    _log("measurement only: controller.py untouched, no threshold fitted, nothing re-decided")
    _rule("=")

    results = []
    for name, audit, tx, sol in D.BATCHES:
        path = os.path.join(data_dir, audit)
        if not os.path.exists(path):
            _log(f"\n[skip] {name}: {audit} not found")
            continue
        df, _stats = D.load_batch(data_dir, audit, tx, sol)
        if not len(df):
            _log(f"\n[skip] {name}: no auto-closed rows")
            continue
        with open(path, encoding="utf-8") as fh:
            n_all = sum(1 for _ in fh)

        section_structure(df, name)
        section_concentration(df)
        cov0, prec0, rows = section_pricing(df, n_all)
        section_residue(df)
        v = section_verdict(df, n_all, cov0, prec0, rows, name)
        if v:
            results.append(v)

    if len(results) > 1:
        _log()
        _rule("=")
        _log("ACROSS BOTH BATCHES")
        _rule("=")
        _log()
        _log(f"  {'batch':<26}{'recall':>9}{'lift':>8}{'pts/pt':>9}{'d coverage':>12}"
             f"{'d precision':>13}{'verdict':>10}")
        _log("  " + "-" * 85)
        for r in results:
            _log(f"  {r['batch']:<26}{r['recall'] * 100:>8.2f}%{r['lift']:>7.2f}x"
                 f"{r['rate']:>9.1f}{r['d_cov']:>+11.3f}{r['d_pre']:>+12.3f}"
                 f"{r['verdict']:>10}")
        _log()
        _log("  BenchRec eval is the in-domain batch and the source of the headline")
        _log("  figures; the synthetic batch is generated data whose pool structure this")
        _log("  project chose, so it can confirm a mechanism but cannot settle a price.")
        _log("  Where the two disagree about affordability, eval governs.")

    _log()
    _rule("=")
    _log("done")
    _rule("=")


if __name__ == "__main__":
    main()
