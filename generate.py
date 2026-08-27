"""
generate.py — synthetic reconciliation data in the BenchRec cash schema.

    generate(n_groups, seed, config) -> {"transactions", "solution", "manifest"}

Emits three files:
    synth_transactions.csv   27-column long format, labels blank, exactly like eval
    synth_solution.csv       B_id, targetAllocation, Usage
    synth_manifest.json      config, seed, and the true class of every group

retrieve.py, complete.py and score.py are NOT modified. score.score and
score._parse_alloc are imported for validation; retrieve is imported for the
retriever baseline.

Schema fidelity (verified against BenchRec_cash_v1.0_eval.csv):
  * A_allocation == f"{currency}_{A_valueDate}_{account}_{A_transactionAttributes}",
    which holds for 100% of real A rows.
  * Each row is ONE transaction: A_* populated and B_* blank, or the mirror.
  * matchId / matchDate / matchRule / matchedBy / targetAllocation are BLANK on every
    row, as they are in eval. Group membership is ground truth — writing matchId into
    the transactions file would leak the allocation set, so it lives in the manifest.
  * wasPreviouslyMismatched is "0" throughout, as in eval.
  * importDate = valueDate + 1 day.
  * DR/CR: the two sides use opposite sign conventions (A: CR->positive, DR->negative;
    B: CR->negative, DR->positive), so a matched pair carries the SAME raw sign with
    FLIPPED flags. Reproduced exactly.
  * The linking signal is a shared ~9-digit run embedded in B_transactionAttributes and
    in the A-side reference/attribute text, surrounded by unrelated noise. B's own
    B_transactionReferences deliberately shares nothing with the A side, as in the real
    data.

All money is handled as integer cents and only formatted to 2dp on output.

Run:  python generate.py [outdir]
"""

from __future__ import annotations

import json
import os
import sys
import time
import copy

import numpy as np
import pandas as pd

from score import score, _parse_alloc

COLUMNS = ['matchId', 'matchDate', 'matchRule', 'matchedBy', 'wasPreviouslyMismatched',
           'A_transactionType', 'A_id', 'A_allocation', 'A_importDate', 'A_debitOrCredit',
           'A_amount', 'A_valueDate', 'A_currencyCode', 'A_account',
           'A_transactionReferences', 'A_transactionAttributes',
           'B_transactionType', 'B_id', 'B_importDate', 'B_debitOrCredit', 'B_amount',
           'B_valueDate', 'B_currencyCode', 'B_account', 'B_transactionReferences',
           'B_transactionAttributes', 'targetAllocation']

WORDS = ["SMILAX", "BOLL", "MORIBUND", "VEXES", "WHUFF", "FOMALHAUT", "ETHERIFIES",
         "SLATTERN", "AVIARY", "ENNOBLERS", "TONITE", "REAVOWED", "RODINESQUE",
         "GULE", "LAROID", "ARATORY", "PILER", "FAMES", "TIRO", "HAU", "OKET",
         "NAPE", "JAUP", "ARGAS", "CACK", "VOLERY", "JOQGE", "TUP", "IFV", "ECCA"]
CODES = ["NAPE", "JAUP", "ARGAS", "OKET", "CACK", "TOM", "IIT", "NUMB", "BRAV"]

# ----------------------------------------------------------------------------------
# Configuration. Every structure class and every corruption has its own rate.
# ----------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Structure classes are mutually exclusive per group and are normalised to sum
    # to 1. Defaults mirror the real eval mix (~93.8% single-key, ~5.6% multi-key,
    # ~0.7% blank), with the multi-key mass split ~70/25/5 repeat/neither/partition
    # as measured on real train data.
    "structure_rates": {
        "one_to_one":  0.9370,
        "repeat":      0.0390,   # ~70% of multi-key
        "neither":     0.0140,   # ~25% of multi-key
        "partition":   0.0030,   # ~5%  of multi-key  (~3% as requested)
        "unmatchable": 0.0070,
    },
    # Independent corruptions, applied on top of whatever structure a group has.
    "corruption_rates": {
        "fee_deduction":       0.10,
        "timing_offset":       0.25,
        "duplicate_reference": 0.08,
    },
    "fee": {
        "fixed_prob": 0.5,             # else percentage
        "fixed_cents": [100, 500000],
        "pct_range": [0.0005, 0.02],
    },
    "timing": {"min_days": 1, "max_days": 7},
    "duplicate_reference": {
        # "distractor": emit an extra ungrouped A row reusing a group A row's
        #               reference (does not perturb the label)
        # "in_group":   reuse the reference across two A rows inside the group
        "scope": "distractor",
        # If True the duplicate also carries the same amount, making it a near-clone.
        # That is what makes this corruption bite against an amount-blocked matcher —
        # with a different amount the block filters it out and the corruption becomes a
        # no-op. The cost is that it then also acts as a same-amount distractor, so the
        # two effects are conflated. Set False to test text confusion in isolation.
        "same_amount": True,
    },
    # Right-skewed group sizes: mostly 2, tail out past 10.
    "group_size": {
        "repeat":    {"min": 2, "max": 12, "decay": 0.42},
        "neither":   {"min": 2, "max": 6,  "decay": 0.55},
        "partition": {"min": 2, "max": 5,  "decay": 0.60},
    },
    # Near-miss distractors: plausible but wrong A rows, so trivial matching cannot
    # score 100%. Counted per group, Poisson.
    "distractors": {
        "per_group_lambda": 1.0,
        "same_amount_prob": 0.35,      # copies the group's amount exactly
        "amount_jitter_cents": [1, 20000],
        "date_jitter_days": 7,
    },
    # Amounts are log-uniform in cents, spanning a wide magnitude range like the real
    # data (real |amount| runs from cents up to ~6.6e9).
    "amount": {"log10_cents_min": 3.0, "log10_cents_max": 11.0},
    "dates": {"start": "2023-03-01", "end": "2023-05-31"},
    "currencies": ["USD"],
    "accounts": ["ACC#00001"],
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ----------------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------------
class _Gen:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._ids = set()
        d0 = np.datetime64(cfg["dates"]["start"])
        d1 = np.datetime64(cfg["dates"]["end"])
        self._span = int((d1 - d0) / np.timedelta64(1, "D"))
        self._d0 = d0

    def txn_id(self):
        while True:
            v = str(int(self.rng.integers(10 ** 11, 10 ** 12)))
            if v not in self._ids:
                self._ids.add(v)
                return v

    def date(self):
        return self._d0 + np.timedelta64(int(self.rng.integers(0, self._span + 1)), "D")

    @staticmethod
    def fmt(d):
        return str(np.datetime64(d, "D"))

    def cents(self):
        lo, hi = self.cfg["amount"]["log10_cents_min"], self.cfg["amount"]["log10_cents_max"]
        return int(10 ** self.rng.uniform(lo, hi))

    def word(self, n=1):
        return " ".join(self.rng.choice(WORDS, size=n))

    def core9(self):
        """A 9-digit reference core with leading zeros, as in the real data."""
        return "00" + str(int(self.rng.integers(1_000_000, 9_999_999)))

    def code(self):
        return f"{self.rng.choice(CODES)}{int(self.rng.integers(100, 999))}"

    def skewed_size(self, spec):
        """Right-skewed: geometric-ish decay from `min`, truncated at `max`."""
        lo, hi, decay = spec["min"], spec["max"], spec["decay"]
        k = lo
        while k < hi and self.rng.random() < decay:
            k += 1
        return k

    # --- text ---
    def a_text(self, core, sub5, code):
        ref = (f"{core} {sub5} {code}".ljust(96) + self.word(3))
        blob = f"{int(self.rng.integers(10**9, 10**10))}{core[2:]}{sub5}{code}"
        attr = (f"66912     {blob}         {self.word(2)} SMILAX   BOLL      "
                f"{core}{sub5}{code}         {self.word(2)}")
        return ref, attr

    def b_text(self, core):
        # B_transactionReferences comes from an unrelated numbering system and shares
        # nothing with the A side — as measured on the real data (0/30,057).
        ref = f"{self.rng.choice(WORDS)} {int(self.rng.integers(10**9, 10**10))}FP"
        attr = (f"TUP {int(self.rng.integers(1000, 9999))}{core}"
                f"{int(self.rng.integers(0, 9))}{self.rng.choice(WORDS)}/WAMP/"
                f"{self.rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}")
        return ref, attr


def _blank_row():
    return {c: "" for c in COLUMNS}


def _a_row(g, cur, acct, cents, vdate, ref, attr):
    r = _blank_row()
    dc = "CR" if cents > 0 else "DR"          # A: CR -> positive, DR -> negative
    vd = _Gen.fmt(vdate)
    r.update({
        "wasPreviouslyMismatched": "0",
        "A_transactionType": "A",
        "A_id": g.txn_id(),
        "A_importDate": _Gen.fmt(np.datetime64(vdate) + np.timedelta64(1, "D")),
        "A_debitOrCredit": dc,
        "A_amount": f"{cents / 100:.2f}",
        "A_valueDate": vd,
        "A_currencyCode": cur,
        "A_account": acct,
        "A_transactionReferences": ref,
        "A_transactionAttributes": attr,
    })
    r["A_allocation"] = f"{cur}_{vd}_{acct}_{attr}"
    return r


def _b_row(g, cur, acct, cents, vdate, ref, attr):
    r = _blank_row()
    dc = "DR" if cents > 0 else "CR"          # B: DR -> positive, CR -> negative
    r.update({
        "wasPreviouslyMismatched": "0",
        "B_transactionType": "B",
        "B_id": g.txn_id(),
        "B_importDate": _Gen.fmt(np.datetime64(vdate) + np.timedelta64(1, "D")),
        "B_debitOrCredit": dc,
        "B_amount": f"{cents / 100:.2f}",
        "B_valueDate": _Gen.fmt(vdate),
        "B_currencyCode": cur,
        "B_account": acct,
        "B_transactionReferences": ref,
        "B_transactionAttributes": attr,
    })
    return r


def _encode(keys):
    """Solution encoding, matching the real files exactly: bare for a single key,
    bracketed comma-separated for a set, empty string for unmatchable."""
    keys = list(dict.fromkeys(keys))
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    return "[" + ",".join(keys) + "]"


# ----------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------
def generate(n_groups, seed=0, config=None):
    cfg = _merge(DEFAULT_CONFIG, config)
    g = _Gen(cfg, seed)
    rng = g.rng

    names = list(cfg["structure_rates"])
    probs = np.array([cfg["structure_rates"][k] for k in names], dtype=float)
    probs = probs / probs.sum()

    rows, sol, groups = [], [], []

    for gi in range(int(n_groups)):
        cls = str(rng.choice(names, p=probs))
        match_id = str(int(rng.integers(10 ** 11, 10 ** 12)))
        cur = str(rng.choice(cfg["currencies"]))
        acct = str(rng.choice(cfg["accounts"]))
        b_date = g.date()
        b_cents = g.cents() * (1 if rng.random() < 0.5 else -1)
        core = g.core9()
        b_ref, b_attr = g.b_text(core)
        corruptions = []

        # -- corruption draws (independent) --
        cr = cfg["corruption_rates"]
        fee = (cls == "one_to_one") and rng.random() < cr["fee_deduction"]
        timing = rng.random() < cr["timing_offset"]
        dup_ref = rng.random() < cr["duplicate_reference"]

        def a_date():
            if not timing:
                return b_date
            t = cfg["timing"]
            d = int(rng.integers(t["min_days"], t["max_days"] + 1))
            return np.datetime64(b_date) + np.timedelta64(d * (1 if rng.random() < 0.5 else -1), "D")

        a_rows, keys = [], []

        if cls == "unmatchable":
            pass  # no A counterpart at all; correct label is blank

        elif cls == "one_to_one":
            c = b_cents
            if fee:
                f = cfg["fee"]
                if rng.random() < f["fixed_prob"]:
                    cut = int(rng.integers(f["fixed_cents"][0], f["fixed_cents"][1] + 1))
                else:
                    cut = int(abs(b_cents) * rng.uniform(*f["pct_range"]))
                cut = min(cut, max(abs(b_cents) - 1, 1))
                c = b_cents - cut * (1 if b_cents > 0 else -1)
                corruptions.append("fee_deduction")
            ref, attr = g.a_text(core, f"{int(rng.integers(10000, 99999))}", g.code())
            a_rows.append(_a_row(g, cur, acct, c, a_date(), ref, attr))

        elif cls == "repeat":
            # every A row carries B's EXACT amount, each with a distinct key
            n = g.skewed_size(cfg["group_size"]["repeat"])
            for _ in range(n):
                ref, attr = g.a_text(core, f"{int(rng.integers(10000, 99999))}", g.code())
                a_rows.append(_a_row(g, cur, acct, b_cents, a_date(), ref, attr))

        elif cls == "neither":
            # amounts neither repeat B nor sum to it
            n = g.skewed_size(cfg["group_size"]["neither"])
            for _ in range(n):
                while True:
                    c = int(abs(b_cents) * rng.uniform(0.2, 3.0)) * (1 if b_cents > 0 else -1)
                    if c != b_cents and c != 0:
                        break
                ref, attr = g.a_text(core, f"{int(rng.integers(10000, 99999))}", g.code())
                a_rows.append(_a_row(g, cur, acct, c, a_date(), ref, attr))
            tot = sum(int(round(float(r["A_amount"]) * 100)) for r in a_rows)
            if tot == b_cents:                      # guard: must not accidentally partition
                a_rows[0]["A_amount"] = f"{(int(round(float(a_rows[0]['A_amount']) * 100)) + 7) / 100:.2f}"

        elif cls == "partition":
            n = g.skewed_size(cfg["group_size"]["partition"])
            mag = abs(b_cents)
            if mag < n:
                mag = n
            # Sample cut points directly; arange(1, mag) would allocate an array of
            # length up to 1e11 for large amounts.
            cuts = np.sort(rng.integers(1, mag, size=n - 1))
            parts = np.diff(np.concatenate(([0], cuts, [mag]))).astype(np.int64)
            parts = [int(p) for p in parts if p > 0]
            if not parts:
                parts = [int(mag)]
            sign = 1 if b_cents > 0 else -1
            for p in parts:
                ref, attr = g.a_text(core, f"{int(rng.integers(10000, 99999))}", g.code())
                a_rows.append(_a_row(g, cur, acct, p * sign, a_date(), ref, attr))

        # duplicate reference
        if dup_ref and a_rows:
            si = int(rng.integers(0, len(a_rows)))
            src = a_rows[si]
            if cfg["duplicate_reference"]["scope"] == "in_group" and len(a_rows) >= 2:
                tgt = a_rows[(si + 1) % len(a_rows)]
                tgt["A_transactionReferences"] = src["A_transactionReferences"]
                corruptions.append("duplicate_reference")
            else:
                src_cents = int(round(float(src["A_amount"]) * 100))
                if not cfg["duplicate_reference"]["same_amount"]:
                    j = int(rng.integers(*cfg["distractors"]["amount_jitter_cents"]))
                    src_cents += j * (1 if rng.random() < 0.5 else -1)
                d = _a_row(g, cur, acct, src_cents,
                           np.datetime64(src["A_valueDate"]),
                           src["A_transactionReferences"],
                           src["A_transactionAttributes"] + " " + g.word(1))
                rows.append(d)                      # ungrouped: label unaffected
                corruptions.append("duplicate_reference")

        if timing and a_rows:
            corruptions.append("timing_offset")

        keys = list(dict.fromkeys(r["A_allocation"] for r in a_rows))
        rows.extend(a_rows)

        b = _b_row(g, cur, acct, b_cents, b_date, b_ref, b_attr)
        rows.append(b)
        sol.append({"B_id": b["B_id"], "targetAllocation": _encode(keys), "Usage": "Public"})

        # -- near-miss distractors: plausible, wrong, ungrouped --
        dcfg = cfg["distractors"]
        n_d = int(rng.poisson(dcfg["per_group_lambda"]))
        n_same = 0
        for _ in range(n_d):
            if rng.random() < dcfg["same_amount_prob"]:
                dc_cents = b_cents
                n_same += 1
            else:
                j = int(rng.integers(*dcfg["amount_jitter_cents"]))
                dc_cents = b_cents + j * (1 if rng.random() < 0.5 else -1)
            dd = np.datetime64(b_date) + np.timedelta64(
                int(rng.integers(-dcfg["date_jitter_days"], dcfg["date_jitter_days"] + 1)), "D")
            ref, attr = g.a_text(g.core9(), f"{int(rng.integers(10000, 99999))}", g.code())
            rows.append(_a_row(g, cur, acct, dc_cents, dd, ref, attr))

        groups.append({
            "matchId": match_id, "class": cls, "corruptions": sorted(set(corruptions)),
            "b_ids": [b["B_id"]], "a_ids": [r["A_id"] for r in a_rows],
            "n_keys": len(keys), "n_distractors": n_d,
            "n_same_amount_distractors": n_same,
        })

    tx = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    tx = tx.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)  # shuffle
    solution = pd.DataFrame(sol, columns=["B_id", "targetAllocation", "Usage"])

    manifest = {
        "seed": int(seed), "n_groups": int(n_groups),
        "n_rows": int(len(tx)),
        "n_a_rows": int((tx["A_transactionType"] == "A").sum()),
        "n_b_rows": int((tx["B_transactionType"] == "B").sum()),
        "config": cfg,
        "class_counts": pd.Series([g["class"] for g in groups]).value_counts().to_dict(),
        "groups": groups,
    }
    return {"transactions": tx, "solution": solution, "manifest": manifest}


def write(result, outdir, prefix="synth"):
    tp = os.path.join(outdir, f"{prefix}_transactions.csv")
    sp = os.path.join(outdir, f"{prefix}_solution.csv")
    mp = os.path.join(outdir, f"{prefix}_manifest.json")
    result["transactions"].to_csv(tp, index=False)
    result["solution"].to_csv(sp, index=False)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(result["manifest"], fh, indent=1)
    return tp, sp, mp


# ----------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------
def baseline_exact_amount(tx, window_days=7):
    """Strongest trivially-simple baseline: for each B row take every A row in the same
    currency/account and date window whose amount matches EXACTLY, and emit the set of
    their distinct allocation keys. Blank when nothing matches."""
    a = tx[tx["A_transactionType"] == "A"]
    b = tx[tx["B_transactionType"] == "B"]
    ac = np.round(pd.to_numeric(a["A_amount"]) * 100).astype(np.int64).to_numpy()
    bc = np.round(pd.to_numeric(b["B_amount"]) * 100).astype(np.int64).to_numpy()
    ad = pd.to_datetime(a["A_valueDate"]).to_numpy()
    bd = pd.to_datetime(b["B_valueDate"]).to_numpy()
    ak = (a["A_currencyCode"] + "|" + a["A_account"]).to_numpy()
    bk = (b["B_currencyCode"] + "|" + b["B_account"]).to_numpy()
    alloc = a["A_allocation"].to_numpy()
    w = np.timedelta64(window_days, "D")

    index = {}
    for i in range(len(ac)):
        index.setdefault((ak[i], ac[i]), []).append(i)

    out = []
    for j in range(len(bc)):
        hits = index.get((bk[j], bc[j]), [])
        keys = [alloc[i] for i in hits if abs(ad[i] - bd[j]) <= w]
        out.append(_encode(keys))
    return pd.DataFrame({"B_id": b["B_id"].to_numpy(), "targetAllocation": out})


def baseline_retriever(tx_path):
    """The existing retriever recipe, imported unmodified."""
    import retrieve as R
    a, b = R._load_sides(tx_path)
    preds, _, _, _ = R.run_all(a, b)
    return preds["cosine+amount"][["B_id", "targetAllocation"]]


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------
def _per_class(pred, sol_path, manifest):
    labels = {r.B_id: _parse_alloc(r.targetAllocation)
              for r in pd.read_csv(sol_path, dtype=str, keep_default_na=False).itertuples()}
    cls_of, corr_of = {}, {}
    for grp in manifest["groups"]:
        for bid in grp["b_ids"]:
            cls_of[bid] = grp["class"]
            corr_of[bid] = grp["corruptions"]

    recs = []
    for r in pred.itertuples():
        p = _parse_alloc(r.targetAllocation)
        gold = labels.get(str(r.B_id), set())
        recs.append({"cls": cls_of.get(str(r.B_id), "?"),
                     "corr": corr_of.get(str(r.B_id), []),
                     "correct": p == gold, "predicted": len(p) > 0})
    d = pd.DataFrame(recs)

    rows = []
    for cls, grp in d.groupby("cls"):
        pr = grp["predicted"].sum()
        rows.append({"class": cls, "rows": len(grp),
                     "match_%": round(grp["correct"].mean() * 100, 3),
                     "precision_%": (round(grp[grp["predicted"]]["correct"].mean() * 100, 3)
                                     if pr else None)})
    # corruption cuts are not exclusive, so report them separately
    for name in ("fee_deduction", "timing_offset", "duplicate_reference"):
        sel = d[d["corr"].map(lambda c: name in c)]
        if len(sel):
            pr = sel["predicted"].sum()
            rows.append({"class": f"[corruption] {name}", "rows": len(sel),
                         "match_%": round(sel["correct"].mean() * 100, 3),
                         "precision_%": (round(sel[sel["predicted"]]["correct"].mean() * 100, 3)
                                         if pr else None)})
    return pd.DataFrame(rows)


def validate(tx_path, sol_path, manifest, run_retriever=True):
    print()
    print("=" * 96)
    print("VALIDATION")
    print("=" * 96)
    tx = pd.read_csv(tx_path, dtype=str, keep_default_na=False)

    results = {}
    print("\n--- baseline 1: exact amount matching ---")
    p1 = baseline_exact_amount(tx)
    r1 = score(p1, sol_path)
    results["exact_amount"] = r1
    print(f"  overall match {r1['match_rate'] * 100:.3f}%   "
          f"precision {r1['match_precision'] * 100:.3f}%   "
          f"abstain {r1['abstention_rate'] * 100:.3f}%")
    print()
    print(_per_class(p1, sol_path, manifest).to_string(index=False))

    if run_retriever:
        print("\n--- baseline 2: existing retriever recipe (cosine + amount) ---")
        try:
            p2 = baseline_retriever(tx_path)
            r2 = score(p2, sol_path)
            results["retriever"] = r2
            print(f"  overall match {r2['match_rate'] * 100:.3f}%   "
                  f"precision {r2['match_precision'] * 100:.3f}%   "
                  f"abstain {r2['abstention_rate'] * 100:.3f}%")
            print()
            print(_per_class(p2, sol_path, manifest).to_string(index=False))
        except Exception as e:
            print(f"  retriever baseline could not run: {type(e).__name__}: {e}")

    print()
    print("-" * 96)
    print("DIFFICULTY GATE — is exact amount matching above 90%?")
    print("-" * 96)
    em = r1["match_rate"] * 100
    print(f"\n  exact amount matching overall match rate: {em:.3f}%")
    if em > 90.0:
        print("\n  >>> TOO EASY. Exact amount matching clears 90%, which means the data is")
        print("  >>> solved by a one-line rule and is not worth shipping as a benchmark.")
        print("  >>> Raise distractors.per_group_lambda / distractors.same_amount_prob, or")
        print("  >>> raise corruption_rates, and regenerate.")
    else:
        print(f"\n  Below the 90% gate by {90.0 - em:.3f} points — exact amount matching does")
        print("  not solve this data, so the set is worth keeping.")
    return results


def _main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    for n, prefix, run_ret in [(50, "synth_small", True), (50_000, "synth", True)]:
        print()
        print("#" * 96)
        print(f"# generate(n_groups={n:,}, seed=7)  ->  prefix '{prefix}'")
        print("#" * 96)
        t0 = time.perf_counter()
        res = generate(n_groups=n, seed=7)
        tp, sp, mp = write(res, outdir, prefix=prefix)
        m = res["manifest"]
        print(f"\n  rows {m['n_rows']:,}  (A {m['n_a_rows']:,}, B {m['n_b_rows']:,})   "
              f"generated in {time.perf_counter() - t0:.2f}s")
        print(f"  class counts: {m['class_counts']}")
        print(f"  files: {os.path.basename(tp)}, {os.path.basename(sp)}, "
              f"{os.path.basename(mp)}")
        validate(tp, sp, m, run_retriever=run_ret)


if __name__ == "__main__":
    _main()
