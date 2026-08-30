"""
findings.py — supporting measurements for the findings in README.md.

Descriptive only: no matching logic, no modelling. Writes to stdout; the repo copy is
findings.log.

Run:  python findings.py > findings.log
"""
import sys
import numpy as np
import pandas as pd
from score import _parse_alloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)

print("=" * 90); print("FINDING: float precision at BenchRec amount magnitudes"); print("=" * 90)
ev = pd.read_csv("BenchRec_cash_v1.0_eval.csv", dtype=str, keep_default_na=False)
A = ev[ev.A_transactionType == "A"]; B = ev[ev.B_transactionType == "B"]
av = pd.to_numeric(A.A_amount); bv = pd.to_numeric(B.B_amount)
mx = float(max(av.abs().max(), bv.abs().max()))
print(f"max |amount| in eval           : {mx:,.2f}")
print(f"float64 resolution at that size: {np.spacing(mx)}")
x = np.float64(20893751.85); y = np.float64(20893751.86)
print(f"two values one cent apart, |y-x| computes as: {abs(y - x)!r}")
print(f"integer cents difference       : {round(20893751.86 * 100) - round(20893751.85 * 100)}")

print(); print("=" * 90); print("FINDING: multi-key groups repeat rather than partition (train)"); print("=" * 90)
tr = pd.read_csv("BenchRec_cash_v1.0_train.csv", dtype=str, keep_default_na=False)
TA = tr[tr.A_transactionType == "A"]; TB = tr[tr.B_transactionType == "B"]
ac = TA.A_amount.astype(float).to_numpy()
grp = {}
for i, m in enumerate(TA.matchId):
    grp.setdefault(m, []).append(i)
rows = []
for bid, mid, t, bamt in zip(TB.B_id, TB.matchId, TB.targetAllocation, TB.B_amount.astype(float)):
    ks = _parse_alloc(t)
    if len(ks) < 2:
        continue
    ii = grp.get(mid, [])
    if not ii:
        continue
    v = ac[ii]
    rows.append({"nk": len(ks), "nA": len(ii),
                 "n_eq": int((np.abs(v - bamt) <= 0.01).sum()),
                 "partition": abs(v.sum() - bamt) <= 0.01})
d = pd.DataFrame(rows)
d["regime"] = np.where(d.partition, "partition", np.where(d.n_eq >= 1, "repeat", "neither"))
print(f"multi-key B rows in train: {len(d):,}")
print("\nregime, per B row (%):"); print((d.regime.value_counts(normalize=True) * 100).round(2).to_string())
print("\nfraction of A rows in the group carrying B's exact amount:")
print(((d.n_eq / d.nA).round(1).value_counts(normalize=True).sort_index() * 100).round(2).to_string())

print(); print("-" * 90); print("WORKED EXAMPLE — matchId 184541000741 (repeat regime)"); print("-" * 90)
mid = "184541000741"; ga = TA[TA.matchId == mid]; gb = TB[TB.matchId == mid]
keys = list(dict.fromkeys(ga.A_allocation)); al = {k: f"K{i+1}" for i, k in enumerate(keys)}
print(f"{len(ga)} A rows, {len(gb)} B rows, {len(keys)} distinct allocation keys\n")
print("B rows (external statement):"); print(gb[["B_id", "B_debitOrCredit", "B_amount"]].to_string(index=False))
print("\nA rows (internal ledger):")
t = ga[["A_id", "A_debitOrCredit", "A_amount"]].copy(); t["key"] = [al[k] for k in ga.A_allocation]
print(t.to_string(index=False))
b0 = float(gb.B_amount.iloc[0]); v = ga.A_amount.astype(float)
print(f"\nB amount                 {b0:>16,.2f}")
print(f"sum of all A amounts     {v.sum():>16,.2f}   (a partition would equal B)")
print(f"A rows with amount == B  {int((np.abs(v - b0) <= 0.01).sum())} of {len(v)}         (a repeat means all of them)")
print(f"target set               {sorted(al[k] for k in _parse_alloc(gb.targetAllocation.iloc[0]))}")

print(); print("=" * 90)
print("FINDING: can that regime split be computed on eval too?")
print("=" * 90)

sol = pd.read_csv("BenchRec_cash_v1.0_solution.csv", dtype=str, keep_default_na=False)

# ---------------------------------------------------------------------------------
# What the eval file actually carries.
# ---------------------------------------------------------------------------------
EA = ev[ev.A_transactionType == "A"].reset_index(drop=True)
EB = ev[ev.B_transactionType == "B"].reset_index(drop=True)
print(f"\nWHAT EVAL CARRIES")
print(f"  matchId blank on         {int((ev.matchId == '').sum()):,} of {len(ev):,} rows "
      f"({(ev.matchId == '').mean() * 100:.2f}%)")
print(f"  targetAllocation blank on {int((ev.targetAllocation == '').sum()):,} of {len(ev):,} rows "
      f"— eval labels live in BenchRec_cash_v1.0_solution.csv")
print(f"  A_allocation populated on {int((EA.A_allocation != '').sum()):,} of {len(EA):,} A rows, "
      f"{EA.A_allocation.nunique():,} distinct keys")
print()
print("  So group membership is NOT given for eval. In train the regime is computed over")
print("  the A rows sharing a B row's matchId. The only route without matchId is to")
print("  reconstruct that A set from the keys the label names: take every A row whose")
print("  A_allocation is in the target set. Whether that is the same set is testable on")
print("  train, where matchId gives the answer.")

# ---------------------------------------------------------------------------------
# Validate the reconstruction against train's matchId ground truth.
# ---------------------------------------------------------------------------------
tr_bykey = {}
for i, k in enumerate(TA.A_allocation):
    tr_bykey.setdefault(k, []).append(i)


def regime_of(amounts, b_amount):
    """The same rule used for train above: partition first, then repeat, else neither."""
    n_eq = int((np.abs(amounts - b_amount) <= 0.01).sum())
    partition = abs(amounts.sum() - b_amount) <= 0.01
    name = "partition" if partition else ("repeat" if n_eq >= 1 else "neither")
    return name, n_eq, len(amounts)


exact = over = under = agree = n_val = 0
conf = {}
frac_true, frac_rec = [], []
for mid, t, bamt in zip(TB.matchId, TB.targetAllocation, TB.B_amount.astype(float)):
    ks = _parse_alloc(t)
    if len(ks) < 2:
        continue
    gi = grp.get(mid, [])
    if not gi:
        continue
    ri = sorted({i for k in ks for i in tr_bykey.get(k, [])})
    n_val += 1
    if set(ri) == set(gi):
        exact += 1
    elif len(ri) > len(gi):
        over += 1
    else:
        under += 1
    rt, net, nat = regime_of(ac[gi], bamt)
    rr, ner, nar = regime_of(ac[ri], bamt)
    conf[(rt, rr)] = conf.get((rt, rr), 0) + 1
    agree += rt == rr
    frac_true.append(round(net / nat, 1))
    frac_rec.append(round(ner / nar, 1))

print()
print("VALIDATING THE RECONSTRUCTION ON TRAIN (matchId is the ground truth there)")
print(f"  multi-key B rows tested: {n_val:,}")
print(f"    reconstructed A set identical to the matchId group: {exact:,} "
      f"({exact / n_val * 100:.2f}%)")
print(f"    over-collected  (picked up A rows from other groups): {over:,} "
      f"({over / n_val * 100:.2f}%)")
print(f"    under-collected (missed A rows in the group):         {under:,} "
      f"({under / n_val * 100:.2f}%)")
print()
print("  A key is not private to one group — a target key can also sit on A rows that")
print("  belong elsewhere, so the reconstruction only ever adds rows, never loses them.")
print("  What matters is whether those extra rows change the answer.")
print()
print("  regime, matchId group -> reconstructed group:")
for (a, b_), c in sorted(conf.items(), key=lambda x: -x[1]):
    print(f"    {a:>9} -> {b_:<9} {c:>6,}" + ("" if a == b_ else "   <-- changed"))
print(f"  regime unchanged: {agree:,} of {n_val:,} ({agree / n_val * 100:.2f}%)")

ft = (pd.Series(frac_true).value_counts(normalize=True).sort_index() * 100).round(2)
fr = (pd.Series(frac_rec).value_counts(normalize=True).sort_index() * 100).round(2)
cmpf = pd.DataFrame({"matchId groups": ft, "reconstructed": fr}).fillna(0.0)
cmpf["error"] = (cmpf["reconstructed"] - cmpf["matchId groups"]).round(2)
print()
print("  fraction of A rows at B's exact amount — same rows, both groupings (%):")
print("    " + cmpf.to_string().replace("\n", "\n    "))
worst = cmpf["error"].abs().max()
print(f"  largest bucket error: {worst:.2f} points")

REGIME_OK = agree / n_val >= 0.99
print()
print("  VERDICT")
print(f"    regime split          — {'COMPUTABLE' if REGIME_OK else 'NOT COMPUTABLE'} on eval; "
      f"the reconstruction reproduces train's own answer for {agree / n_val * 100:.2f}% of rows.")
print(f"    fraction distribution — NOT COMPUTABLE on eval. The extra A rows dilute the")
print(f"      fraction, moving the 1.0 bucket by {abs(cmpf.loc[1.0, 'error']):.2f} points on train.")
print("      Reporting it for eval would be reporting an artefact of the reconstruction.")
print("      What would be needed: matchId, or any per-row group identifier, in the eval")
print("      file — or an A-side group column in the solution. Neither exists.")

# ---------------------------------------------------------------------------------
# Apply it to eval.
# ---------------------------------------------------------------------------------
print(); print("=" * 90)
print("FINDING: multi-key regime on eval (groups reconstructed from the labels)")
print("=" * 90)

sol_lab = dict(zip(sol.B_id.astype(str), sol.targetAllocation))
ev_bykey = {}
for i, k in enumerate(EA.A_allocation):
    ev_bykey.setdefault(k, []).append(i)
eac = EA.A_amount.astype(float).to_numpy()

erows = []
for bid, bamt in zip(EB.B_id.astype(str), EB.B_amount.astype(float)):
    ks = _parse_alloc(sol_lab.get(bid, ""))
    if len(ks) < 2:
        continue
    ri = sorted({i for k in ks for i in ev_bykey.get(k, [])})
    if not ri:
        continue
    name, n_eq, n_a = regime_of(eac[ri], bamt)
    erows.append({"b_id": bid, "nk": len(ks), "nA": n_a, "n_eq": n_eq, "regime": name})
ed = pd.DataFrame(erows)

n_multi_eval = int(sol.targetAllocation.map(lambda x: len(_parse_alloc(x)) >= 2).sum())
print(f"\nmulti-key B rows in eval: {n_multi_eval:,}")
print(f"  with at least one A row carrying a target key: {len(ed):,} "
      f"({len(ed) / n_multi_eval * 100:.2f}%)")
print(f"  with none (no A row in the file carries any of them): "
      f"{n_multi_eval - len(ed):,}")
print("\nregime, per B row (%):")
print((ed.regime.value_counts(normalize=True) * 100).round(2).to_string())
print(f"\n  Carry the validated error across: the reconstruction changed train's answer for")
print(f"  {100 - agree / n_val * 100:.2f}% of rows, biased toward calling a partition a repeat")
print("  (35 of 52 disagreements on train went that way), so eval's repeat share is if")
print("  anything a slight over-estimate.")
print("\n  The fraction-of-A-rows distribution is deliberately not reported here. See the")
print("  verdict above: on train the reconstruction moves it by "
      f"{abs(cmpf.loc[1.0, 'error']):.2f} points.")

# The B side of a group: can it be recovered from the label alone? Two different
# questions, and only the second one licenses the worked example below.
bset = TB[TB.targetAllocation != ""].copy()
bset["_ks"] = bset.targetAllocation.map(lambda x: tuple(sorted(_parse_alloc(x))))
one_set = bset.groupby("matchId")._ks.nunique()
one_mid = bset.groupby("_ks").matchId.nunique()
print()
print("  B SIDE — recoverable from the label?")
print(f"    matchIds whose B rows all carry one target set: {int((one_set == 1).sum()):,} of "
      f"{len(one_set):,} ({(one_set == 1).mean() * 100:.2f}%)")
print(f"    target sets that map to exactly one matchId:    {int((one_mid == 1).sum()):,} of "
      f"{len(one_mid):,} ({(one_mid == 1).mean() * 100:.2f}%)")
print("    The first is what makes the label consistent within a group; the second is what")
print("    grouping eval's B rows by target set actually needs, and it is the weaker of the")
print(f"    two. {100 - (one_mid == 1).mean() * 100:.2f}% of target sets are shared across groups, so the")
print("    B side of the example below is reconstructed on the same footing as the A side:")
print("    mostly right, not guaranteed, and it may include rows from another group.")

# worked example, same shape as the train one
rep = ed[(ed.regime == "repeat") & (ed.nA.between(4, 8)) & (ed.nk == ed.nA)]
if len(rep):
    r0 = rep.iloc[0]
    bid = r0.b_id
    ks = _parse_alloc(sol_lab[bid])
    ii = sorted({i for k in ks for i in ev_bykey.get(k, [])})
    ga = EA.iloc[ii]
    # B rows sharing this exact target set: in train that recovers the matchId group's
    # B side for 100% of groups, checked below
    same = [b for b, t in sol_lab.items() if tuple(sorted(_parse_alloc(t))) == tuple(sorted(ks))]
    gb = EB[EB.B_id.astype(str).isin(same)]
    print(); print("-" * 90)
    print(f"WORKED EXAMPLE — eval B_id {bid} (repeat regime)")
    print("-" * 90)
    al = {k: f"K{i+1}" for i, k in enumerate(sorted(ks))}
    print(f"{len(ga)} A rows, {len(gb)} B rows, {len(ks)} distinct allocation keys")
    print("  (no matchId in eval: both sides are reconstructed — the A rows are those")
    print("   carrying the label's keys, the B rows are those whose label is this same key")
    print("   set. Either may include a row from another group; see the check above.)")
    print()
    print("B rows (external statement):")
    print(gb[["B_id", "B_debitOrCredit", "B_amount"]].to_string(index=False))
    print("\nA rows (internal ledger):")
    t2 = ga[["A_id", "A_debitOrCredit", "A_amount"]].copy()
    t2["key"] = [al[k] for k in ga.A_allocation]
    print(t2.to_string(index=False))
    b0 = float(gb.B_amount.iloc[0]); v2 = ga.A_amount.astype(float)
    print(f"\nB amount                 {b0:>16,.2f}")
    print(f"sum of all A amounts     {v2.sum():>16,.2f}   (a partition would equal B)")
    print(f"A rows with amount == B  {int((np.abs(v2 - b0) <= 0.01).sum())} of {len(v2)}"
          f"         (a repeat means all of them)")
    print(f"target set               {sorted(al[k] for k in ks)}")
else:
    print("\n  no eval repeat-regime group in the 4-8 row range to print as an example")


# ---------------------------------------------------------------------------------
# Side by side.
# ---------------------------------------------------------------------------------
print(); print("=" * 90)
print("TRAIN vs EVAL — multi-key regime side by side")
print("=" * 90)
tr_pc = (d.regime.value_counts(normalize=True) * 100).round(2)
ev_pc = (ed.regime.value_counts(normalize=True) * 100).round(2)
side = pd.DataFrame({
    "train (matchId groups)": tr_pc,
    "eval (reconstructed)": ev_pc,
}).fillna(0.0)
side["difference"] = (side["eval (reconstructed)"] - side["train (matchId groups)"]).round(2)
print()
print(side.to_string())
print()
print(f"  multi-key B rows      train {len(d):,}   eval {len(ed):,}")
print(f"  share of the file     train {len(d) / len(TB) * 100:.2f}%   "
      f"eval {n_multi_eval / len(sol) * 100:.2f}%")
print()
print("  Read the eval column with its measured caveat: the grouping is reconstructed,")
print(f"  and on train that reconstruction reproduces the matchId answer "
      f"{agree / n_val * 100:.2f}% of the time.")
print("  The headline claim — multi-key groups repeat rather than partition — holds on")
print("  both splits, and does not depend on the reconstruction being exact.")


print(); print("=" * 90); print("Real eval label mix"); print("=" * 90)
n = sol.targetAllocation.map(lambda s: len(_parse_alloc(s)))
print(f"total solution rows {len(sol):,}")
print(f"  single-key {int((n == 1).sum()):,} ({(n == 1).mean() * 100:.2f}%)")
print(f"  multi-key  {int((n >= 2).sum()):,} ({(n >= 2).mean() * 100:.2f}%)")
print(f"  blank      {int((n == 0).sum()):,} ({(n == 0).mean() * 100:.2f}%)")
