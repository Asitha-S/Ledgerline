"""
drift.py — signed balance drift over decisions already made.

Sibling of exposure.py. Reads the audit records controller.py wrote and the solution
files, and asks a narrower question than exposure: not "how much value did a wrong
decision put at risk" but "in which direction, and by how much, does the ledger side of
the posting differ from the ledger side the label names".

It tunes nothing, changes nothing, re-decides nothing and re-ranks nothing. Every
decision and every answer is taken verbatim from the audit.

------------------------------------------------------------------------------------
THE DEFINITION, AND WHERE IT STOPS BEING DEFINABLE

A posting attaches one statement row (B) to a set of ledger allocation keys. The
"amount posted" is therefore the ledger value the posting attaches:

    posted_ledger = sum of the amounts of the keys the controller answered
    gold_ledger   = sum of the amounts of the keys the label names
    drift         = posted_ledger - gold_ledger

SIGN CONVENTION. Amounts are used exactly as the dataset stores them, in integer
cents. In this data a statement debit is positive and a statement credit is negative;
on the ledger side the polarity is inverted (A debits negative, A credits positive), so
a correctly matched pair carries the SAME sign. No amount is negated, abs()'d or
re-polarised anywhere in this script.

    drift > 0   the automation attached MORE ledger value than the label names
    drift < 0   the automation attached LESS ledger value than the label names
    drift = 0   the two key sets carry the same ledger value

Because "more" and "less" in raw sign terms mix statement debits and credits, the
per-row distribution is also reported direction-normalised, as drift * sign(B amount):
positive there means over-attachment in the statement row's own direction.

WHERE THE KEY SETS AGREE, drift is exactly 0 by definition and no amount lookup is
needed: the same key set carries the same value whichever ledger rows it denotes. Only
the symmetric difference of the two key sets can contribute, so only those keys are
priced.

WHERE THE KEY SETS ARE WHOLLY DIFFERENT, "difference" is still the expression above and
it remains computable, but it does NOT mean money lost. Attaching a statement row to
the wrong key is a mis-attribution, not a revaluation: the statement value is posted
either way, one account is overstated and another understated by the same amount, and a
ledger-wide balance check nets to zero by construction. What drift measures is the
value attached to accounts that should not have received it. The full-value figure for
a wrong posting is exposure.py's; this script is about direction and cancellation.

WHERE IT CANNOT BE COMPUTED. Pricing a key requires knowing the amount of the ledger
row it denotes. An allocation key is not unique to a ledger row: on BenchRec eval 2,566
of 22,779 keys (11.26%) are carried by several A rows with DIFFERENT amounts, and the
dataset gives no way to say which one a label meant — matchId, which would identify the
group, is blank on all 69,171 eval rows. For those keys the amount is not determined by
the data. Rather than pick one, this script reports:

    * an EXACT drift where every key in the symmetric difference has one amount, and
    * a BOUNDED drift [min, max] otherwise, from the min and max amount each
      ambiguous key could denote.

Neither is an estimate and no interval is collapsed to a point anywhere below.

WHAT THIS MEASUREMENT IS STRUCTURALLY BLIND TO is reported explicitly, because on this
data it is most of the error: see the final section.

Run:  python drift.py [data_dir]
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import pandas as pd

from score import _parse_alloc     # unmodified — the same parser the scorer uses

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BATCHES = [
    ("BenchRec eval", "controller_audit_eval.jsonl",
     "BenchRec_cash_v1.0_eval.csv", "BenchRec_cash_v1.0_solution.csv"),
    ("synthetic 50,000-group", "controller_audit_synth.jsonl",
     "synth_transactions.csv", "synth_solution.csv"),
]

# |B amount| buckets, in cents. Upper bound exclusive; the last is open.
BUCKETS = [
    ("< 1K",          0,            100_000),
    ("1K - 10K",      100_000,      1_000_000),
    ("10K - 100K",    1_000_000,    10_000_000),
    ("100K - 1M",     10_000_000,   100_000_000),
    ("1M - 10M",      100_000_000,  1_000_000_000),
    ("10M - 100M",    1_000_000_000, 10_000_000_000),
    (">= 100M",       10_000_000_000, None),
]


def _log(m=""):
    print(m.rstrip(), flush=True)


def _rule(c="-", n=104):
    _log(c * n)


def _money(cents) -> str:
    """Cents -> a readable signed magnitude. Values here span cents to billions."""
    d = cents / 100.0
    sign = "-" if d < 0 else ""
    d = abs(d)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if d >= div:
            return f"{sign}{d / div:,.2f}{suf}"
    return f"{sign}{d:,.2f}"


def _pct(n, d):
    return f"{n / d * 100:.3f}%" if d else "n/a"


# ----------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------
def load_batch(data_dir, audit_file, tx_file, solution_file):
    """Returns (rows, keymap_stats). Every amount is integer cents."""
    tx = pd.read_csv(os.path.join(data_dir, tx_file), dtype=str, keep_default_na=False)
    A = tx[tx["A_transactionType"] == "A"]
    a_cents = (pd.to_numeric(A["A_amount"]).astype(float) * 100).round().astype("int64")

    # key -> the set of distinct amounts carried by A rows under that key
    key_amounts = collections.defaultdict(set)
    for k, c in zip(A["A_allocation"], a_cents):
        key_amounts[k].add(int(c))
    # a_id -> key, so a candidate recorded in the audit can be priced exactly
    aid_key = dict(zip(A["A_id"].astype(str), A["A_allocation"]))

    stats = {
        "a_rows": len(A),
        "keys": len(key_amounts),
        "keys_one_amount": sum(1 for v in key_amounts.values() if len(v) == 1),
    }

    sol = pd.read_csv(os.path.join(data_dir, solution_file), dtype=str,
                      keep_default_na=False)
    labels = {str(b): _parse_alloc(t)
              for b, t in zip(sol["B_id"], sol["targetAllocation"])}

    rows = []
    with open(os.path.join(data_dir, audit_file), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["decision"] != "auto_close":
                continue
            bid = str(d["b_id"])
            posted = set(d.get("answer_keys") or [])
            gold = labels.get(bid, set())
            b_cents = int(d["b_amount_cents"])

            # prices for the keys the controller actually posted: the audit names the
            # exact ledger row behind each candidate, so no ambiguity on this side
            cand_price = {}
            for c in (d.get("candidates") or []):
                k = aid_key.get(str(c["a_id"]))
                if k is not None:
                    cand_price.setdefault(k, int(c["amount_cents"]))

            cand_keys = {aid_key.get(str(c["a_id"])) for c in (d.get("candidates") or [])}
            cand_keys.discard(None)

            only_p = posted - gold
            only_g = gold - posted

            lo = hi = 0
            exact = True
            unpriceable = []
            for k in only_p:
                if k in cand_price:
                    lo += cand_price[k]
                    hi += cand_price[k]
                elif k in key_amounts:
                    v = key_amounts[k]
                    lo += min(v)
                    hi += max(v)
                    exact = exact and len(v) == 1
                else:
                    unpriceable.append(k)
            for k in only_g:
                if k in key_amounts:
                    v = key_amounts[k]
                    lo -= max(v)          # subtracting: the max lowers the bound
                    hi -= min(v)
                    exact = exact and len(v) == 1
                else:
                    unpriceable.append(k)

            rows.append({
                "b_id": bid,
                "b_cents": b_cents,
                "abs_b": abs(b_cents),
                "correct": posted == gold,
                "gold_blank": len(gold) == 0,
                "n_posted": len(posted),
                "n_gold": len(gold),
                "stratum": ("blank" if not gold else
                            "single-key" if len(gold) == 1 else "multi-key"),
                "dup_ref": bool(d.get("duplicate_reference_among_candidates")),
                "n_cands": len(d.get("candidates") or []),
                "n_exact_cand": sum(1 for c in (d.get("candidates") or [])
                                    if c.get("exact_amount")),
                # the decision is over KEYS, so candidate rows sharing one key are
                # one choice, not several
                "n_exact_keys": len({aid_key.get(str(c["a_id"]))
                                     for c in (d.get("candidates") or [])
                                     if c.get("exact_amount")} - {None}),
                "pool_size": int(d.get("pool_size") or 0),
                "top1_score": d.get("top1_score"),
                "margin": d.get("margin"),
                "gold_in_pool": bool(gold & cand_keys),
                "exception_class": d.get("exception_class") or "",
                "n_triggers": len(d.get("triggers") or []),
                "drift_lo": lo,
                "drift_hi": hi,
                "exact": exact and not unpriceable,
                "unpriceable": len(unpriceable),
            })
    return pd.DataFrame(rows), stats


# ----------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------
def section_definition(name, df, stats):
    _log()
    _rule("=")
    _log(f"SIGNED BALANCE DRIFT — {name}")
    _rule("=")
    _log()
    _log("  drift  =  (ledger value the controller attached)")
    _log("            - (ledger value the label names)          per auto-closed row, in cents")
    _log()
    _log("  drift > 0   more ledger value attached than the label names")
    _log("  drift < 0   less")
    _log("  drift = 0   equal value, which INCLUDES wrong postings onto equal-valued keys")
    _log()
    _log("  Amounts are used with the sign the dataset stores. A statement debit is")
    _log("  positive, a statement credit negative, and a correctly matched ledger row")
    _log("  carries the same sign as its statement row. Nothing is negated or abs()'d.")
    _log()
    _log("  This is mis-ATTRIBUTION, not loss. The statement value is posted either way;")
    _log("  one account is overstated and another understated. exposure.py reports the")
    _log("  full value at risk. This reports direction and cancellation.")
    _log()

    n = len(df)
    ok = int(df["correct"].sum())
    _log(f"  auto-closed rows measured        {n:,}")
    _log(f"    key sets agree, drift = 0      {ok:,}   ({_pct(ok, n)})")
    _log(f"    key sets differ                {n - ok:,}   ({_pct(n - ok, n)})")
    _log()
    _log(f"  All drift originates in {n - ok:,} of {n:,} decisions. The rest contribute exactly")
    _log("  zero and need no amount lookup: an identical key set carries an identical value")
    _log("  whichever ledger rows it denotes.")

    # structural facts worth stating rather than assuming
    posted_sizes = collections.Counter(df["n_posted"])
    _log()
    _log(f"  keys per posted answer           {dict(sorted(posted_sizes.items()))}")
    if set(posted_sizes) == {1}:
        _log("    Every auto-closed answer is a single key. This is structural, not")
        _log("    incidental: controller.py escalates whenever set completion adds a key")
        _log("    (t_add), so a multi-key answer can never reach auto-close.")
    _log(f"  label stratum of those rows      "
         f"{dict(collections.Counter(df['stratum']).most_common())}")


def section_computability(df, stats):
    _log()
    _rule()
    _log("COMPUTABILITY — where the difference is defined, and where it is not")
    _rule()
    _log()
    amb = stats["keys"] - stats["keys_one_amount"]
    _log(f"  A rows {stats['a_rows']:,}   distinct allocation keys {stats['keys']:,}")
    _log(f"  keys carried by rows of one amount   {stats['keys_one_amount']:,} "
         f"({_pct(stats['keys_one_amount'], stats['keys'])})")
    _log(f"  keys carried by rows of >1 amount    {amb:,} "
         f"({_pct(amb, stats['keys'])})  <- amount not determined by the data")
    _log()

    bad = df[~df["correct"]]
    ex = int(bad["exact"].sum())
    bd = len(bad) - ex
    unp = int((bad["unpriceable"] > 0).sum())
    _log(f"  of the {len(bad):,} rows whose key sets differ:")
    _log(f"    drift EXACT                        {ex:,}   ({_pct(ex, len(bad))})")
    _log(f"    drift BOUNDED only                 {bd:,}   ({_pct(bd, len(bad))})")
    if bd:
        _log("      because a key in the symmetric difference is carried by several")
        _log("      ledger rows of different amounts and matchId is blank")
    _log(f"    no bound at all (key absent from A)  {unp:,}")
    if bd:
        _log()
        w = bad[~bad["exact"]]
        span = (w["drift_hi"] - w["drift_lo"]).abs()
        _log(f"    width of those intervals: median {_money(span.median())}, "
             f"max {_money(span.max())}")
        _log("    No interval is collapsed to a point anywhere in this report.")


def section_headline(df):
    _log()
    _rule()
    _log("NET AND GROSS")
    _rule()
    _log()
    ex = df[df["exact"]]
    inx = df[~df["exact"]]

    net = int(ex["drift_lo"].sum())
    gross = int(ex["drift_lo"].abs().sum())
    _log(f"  On the {len(ex):,} rows where drift is exact "
         f"({len(ex[~ex['correct']]):,} of them non-zero):")
    _log()
    _log(f"    net   signed drift    {_money(net):>18}")
    _log(f"    gross absolute drift  {_money(gross):>18}")
    if gross:
        _log(f"    net / gross           {abs(net) / gross * 100:>17.3f}%"
             "   <- 100% means no cancellation at all")
        _log(f"    cancelled             {_money(gross - abs(net)):>18}")
    _log()

    if len(inx):
        lo = net + int(inx["drift_lo"].sum())
        hi = net + int(inx["drift_hi"].sum())
        _log(f"  Adding the {len(inx):,} bounded rows, the net over ALL "
             f"{len(df):,} auto-closed rows lies in:")
        _log(f"    [ {_money(lo)} , {_money(hi)} ]")
        # gross: a row whose interval straddles zero can contribute as little as 0
        g_lo = gross + int(inx.apply(
            lambda r: 0 if r["drift_lo"] <= 0 <= r["drift_hi"]
            else min(abs(r["drift_lo"]), abs(r["drift_hi"])), axis=1).sum())
        g_hi = gross + int(inx[["drift_lo", "drift_hi"]].abs().max(axis=1).sum())
        _log(f"    gross in  [ {_money(g_lo)} , {_money(g_hi)} ]")
        _log()
    _log("  Read the net as the extent to which automating these decisions moved the")
    _log("  ledger in one direction rather than washing out. It is not a loss figure.")


def section_distribution(df):
    _log()
    _rule()
    _log("DISTRIBUTION OF SIGNED PER-ROW DRIFT   (exact rows only)")
    _rule()
    _log()
    ex = df[df["exact"]]
    d = ex["drift_lo"]
    pos, neg, zer = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    _log(f"  positive (over-attached)   {pos:,}")
    _log(f"  negative (under-attached)  {neg:,}")
    _log(f"  exactly zero               {zer:,}   "
         f"({zer - int(ex['correct'].sum()):,} of them WRONG postings — see the blind spot below)")
    _log()

    for lbl, sub in (("positive", d[d > 0]), ("negative", d[d < 0].abs())):
        if not len(sub):
            _log(f"  {lbl}: none")
            continue
        q = sub.quantile([.5, .9, .99]).astype("int64")
        _log(f"  {lbl} magnitudes  n={len(sub):,}  "
             f"min {_money(sub.min())}  median {_money(q[.5])}  "
             f"p90 {_money(q[.9])}  p99 {_money(q[.99])}  max {_money(sub.max())}  "
             f"sum {_money(sub.sum())}")
    _log()

    # direction-normalised: is the row over- or under-attached in its own direction?
    nz = ex[ex["drift_lo"] != 0]
    if len(nz):
        norm = nz["drift_lo"] * np.sign(nz["b_cents"])
        _log(f"  direction-normalised (drift x sign of the statement amount), {len(nz):,} non-zero rows:")
        _log(f"    over-attached in the row's own direction   {int((norm > 0).sum()):,}")
        _log(f"    under-attached                             {int((norm < 0).sum()):,}")
        _log(f"    net normalised                             {_money(int(norm.sum()))}")
        g = int(norm.abs().sum())
        if g:
            _log(f"    net / gross, normalised                   "
                 f"{abs(int(norm.sum())) / g * 100:.3f}%")
            _log("      Compare this with the raw net/gross above. The raw figure treats a")
            _log("      statement debit and a statement credit as opposite, so the two cancel")
            _log("      in the total even when both are errors in the same direction. Where")
            _log("      the normalised ratio is the higher of the two, the bias is systematic")
            _log("      and the raw net understates it.")
        _log()
        by = nz.assign(norm=norm).groupby("stratum", sort=False)
        _log("    by label stratum:")
        for k, sub in by:
            _log(f"      {str(k):<12} over {int((sub['norm'] > 0).sum()):>5,}   "
                 f"under {int((sub['norm'] < 0).sum()):>5,}   "
                 f"net {_money(int(sub['norm'].sum())):>12}")
        mk = nz.assign(norm=norm)
        mk = mk[mk["stratum"] == "multi-key"]
        if len(mk):
            u = int((mk["norm"] < 0).sum())
            _log(f"      A multi-key label cannot be answered by an auto-close, which posts")
            _log(f"      exactly one key, and {u:,} of {len(mk):,} such rows under-attach "
                 f"({_pct(u, len(mk))}).")
            if u < len(mk):
                _log("      Not all of them: one large ledger row can still exceed the sum of")
                _log("      several small ones, so the direction is empirical, not structural.")
        bl = nz.assign(norm=norm)
        bl = bl[bl["stratum"] == "blank"]
        if len(bl) and (bl["norm"] > 0).all():
            _log("      Blank labels over-attach without exception, and that one IS structural:")
            _log("      the label names no ledger value at all, so drift is the whole posted")
            _log("      amount, which matched the statement to the cent.")


def _group_table(df, by, title, note=None):
    _log()
    _rule()
    _log(title)
    _rule()
    if note:
        _log()
        for ln in note:
            _log("  " + ln)
    _log()
    _log(f"  {'group':<16}{'rows':>9}{'differing':>11}{'exact':>8}"
         f"{'net (exact)':>16}{'gross (exact)':>16}{'bounded net range':>36}")
    _log("  " + "-" * 110)
    for key, sub in df.groupby(by, sort=False):
        ex = sub[sub["exact"]]
        inx = sub[~sub["exact"]]
        net = int(ex["drift_lo"].sum())
        gross = int(ex["drift_lo"].abs().sum())
        rng = ""
        if len(inx):
            rng = (f"[{_money(net + int(inx['drift_lo'].sum()))}, "
                   f"{_money(net + int(inx['drift_hi'].sum()))}]")
        _log(f"  {str(key):<16}{len(sub):>9,}{int((~sub['correct']).sum()):>11,}"
             f"{int(sub['exact'].sum()):>8,}{_money(net):>16}{_money(gross):>16}{rng:>36}")


def section_by_class(df):
    _log()
    _rule()
    _log("BY EXCEPTION CLASS")
    _rule()
    _log()
    classes = collections.Counter(df["exception_class"])
    if set(classes) - {""}:
        _group_table(df, "exception_class", "BY EXCEPTION CLASS")
        return
    _log(f"  NOT COMPUTABLE. exception_class is empty on all {len(df):,} auto-closed rows,")
    _log(f"  and {int((df['n_triggers'] == 0).sum()):,} of them recorded no trigger either.")
    _log()
    _log("  This is by construction, not a gap in the audit. In controller.py the escalate")
    _log("  flag is  esc = t_no | t_add | t_fee | t_low,  and the class is a select over")
    _log("  five branches built from those same four triggers. A row carries a class")
    _log("  exactly when a trigger fires, and escalates exactly when a trigger fires. An")
    _log("  auto-closed row is therefore, by definition, a row with no class. Classes")
    _log("  partition the escalated queue; they do not reach the auto-closed population,")
    _log("  and no breakdown of auto-close drift by exception class exists to be reported.")
    _log()
    _log("  Substituted below are the two axes that DO vary across auto-closed rows: the")
    _log("  stratum of the label, and the one class-adjacent flag the audit records for")
    _log("  rows that were not escalated.")

    _group_table(df, "stratum", "BY LABEL STRATUM   (substitute for exception class)", [
        "single-key / multi-key / blank refers to the LABEL, not the answer — every",
        "auto-closed answer is a single key.",
    ])
    _group_table(df.assign(dup=np.where(df["dup_ref"], "dup ref seen", "no dup ref")),
                 "dup", "BY DUPLICATE-REFERENCE FLAG   (recorded on auto-closed rows)", [
                     "duplicate_reference_among_candidates is recorded for every row but only",
                     "escalates in combination with set completion, so it survives here.",
                 ])


def section_by_magnitude(df):
    def bucket(v):
        for lbl, lo, hi in BUCKETS:
            if v >= lo and (hi is None or v < hi):
                return lbl
        return BUCKETS[-1][0]

    d = df.assign(bucket=df["abs_b"].map(bucket))
    order = [b[0] for b in BUCKETS if (d["bucket"] == b[0]).any()]
    d["bucket"] = pd.Categorical(d["bucket"], categories=order, ordered=True)
    _group_table(d.sort_values("bucket"), "bucket",
                 "BY STATEMENT AMOUNT MAGNITUDE   (|B amount|)",
                 ["Buckets are on the statement row's own amount, so a row appears in the",
                  "bucket a reviewer would sort it into, whatever its drift turned out to be."])


def section_worked_example(df):
    """The largest-magnitude bucket that contains wrong postings, row by row."""
    def bucket_idx(v):
        for i, (lbl, lo, hi) in enumerate(BUCKETS):
            if v >= lo and (hi is None or v < hi):
                return i
        return len(BUCKETS) - 1

    bad = df[~df["correct"]]
    if not len(bad):
        return
    bi = bad["abs_b"].map(bucket_idx)
    top = int(bi.max())
    rows = bad[bi == top].sort_values("abs_b", ascending=False)
    ex = rows[rows["exact"]]

    _log()
    _rule()
    _log(f"WORKED EXAMPLE — the {BUCKETS[top][0]} bucket, every wrong posting in it")
    _rule()
    _log()
    _log(f"  {'B_id':<16}{'|B amount|':>20}{'drift':>20}  label")
    _log("  " + "-" * 76)
    for r in rows.head(10).itertuples():
        d = (_money(r.drift_lo) if r.exact
             else f"[{_money(r.drift_lo)}, {_money(r.drift_hi)}]")
        _log(f"  {r.b_id:<16}{_money(r.abs_b):>20}{d:>20}  {r.stratum}")
    if len(rows) > 10:
        _log(f"  ... and {len(rows) - 10:,} more")
    _log()
    if len(ex):
        net = int(ex["drift_lo"].sum())
        gross = int(ex["drift_lo"].abs().sum())
        _log(f"  net of the exact ones   {_money(net)}")
        _log(f"  gross                   {_money(gross)}")
        _log(f"  value mis-attributed    {_money(int(rows['abs_b'].sum()))}   "
             "(exposure.py's measure, all rows above)")
        if net == 0 and gross:
            _log()
            _log("  The net is exactly zero and the postings are all wrong. Opposite-signed")
            _log("  errors of equal magnitude cancel, and a wrong posting onto an")
            _log("  equal-valued key contributes nothing to begin with. A balance check over")
            _log("  this bucket would report that nothing had happened.")


def section_blind_spot(df):
    _log()
    _rule()
    _log("WHAT THIS MEASUREMENT IS BLIND TO")
    _rule()
    _log()
    bad = df[~df["correct"]]
    ex_bad = bad[bad["exact"]]
    zero = ex_bad[ex_bad["drift_lo"] == 0]
    straddle = bad[(~bad["exact"]) &
                   (bad["drift_lo"] <= 0) & (0 <= bad["drift_hi"])]

    _log(f"  wrong postings with drift EXACTLY zero        {len(zero):,} "
         f"of {len(ex_bad):,} exact wrong rows ({_pct(len(zero), len(ex_bad))})")
    _log(f"  wrong postings whose interval straddles zero  {len(straddle):,} "
         f"of {len(bad) - len(ex_bad):,} bounded wrong rows")
    vis = len(ex_bad) - len(zero)
    _log(f"  wrong postings that DO move value             {vis:,} "
         f"of {len(ex_bad):,} exact wrong rows ({_pct(vis, len(ex_bad))})")
    _log("    — the complement, and the share a balance check would see. A balance")
    _log("      control is partial here, not useless.")
    _log()
    _log(f"  value mis-attributed by those zero-drift rows  "
         f"{_money(int(zero['abs_b'].sum()))}")
    _log("    — invisible to any balance check, because the wrong key carries the same")
    _log("      amount as the right one. The posting is wrong; the totals agree.")
    _log()
    _log("  This is the repeat regime showing up in the drift figures. Where several")
    _log("  ledger rows already carry the statement's exact amount, choosing the wrong")
    _log("  one produces zero drift by construction. A net-drift check cannot find those")
    _log("  errors, and on this data they are not a rounding detail.")
    _log()
    _log("  Also outside this measurement, by construction:")
    _log("    * escalated rows. Drift is defined over postings the system made without")
    _log("      review; an escalated row has not been posted, so it has no drift.")
    _log("    * every auto-closed row matched its top candidate to within 1 cent")
    _log("      (controller.py escalates on t_fee = has & ~exact_top1, TOL_CENTS = 1), so")
    _log("      the posted side of every row here is the statement amount to the cent.")
    _log("      Drift is entirely a property of what the LABEL says, never of a")
    _log("      near-miss on the amount.")


# ----------------------------------------------------------------------------------
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    _log()
    _rule("=")
    _log("drift.py — signed balance drift over decisions already made")
    _log("measurement only: nothing is tuned, re-decided or re-ranked")
    _rule("=")

    for name, audit, tx, sol in BATCHES:
        path = os.path.join(data_dir, audit)
        if not os.path.exists(path):
            _log(f"\n[skip] {name}: {audit} not found")
            continue
        df, stats = load_batch(data_dir, audit, tx, sol)
        if not len(df):
            _log(f"\n[skip] {name}: no auto-closed rows in the audit")
            continue

        # invariant: agreeing key sets must price to exactly zero
        agree = df[df["correct"]]
        assert (agree["drift_lo"] == 0).all() and (agree["drift_hi"] == 0).all(), \
            "a row whose key sets agree priced to non-zero drift"

        section_definition(name, df, stats)
        section_computability(df, stats)
        section_headline(df)
        section_distribution(df)
        section_by_class(df)
        section_by_magnitude(df)
        section_worked_example(df)
        section_blind_spot(df)

    _log()
    _rule("=")
    _log("done")
    _rule("=")


if __name__ == "__main__":
    main()
