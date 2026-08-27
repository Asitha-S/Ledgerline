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

print(); print("=" * 90); print("Real eval label mix"); print("=" * 90)
sol = pd.read_csv("BenchRec_cash_v1.0_solution.csv", dtype=str, keep_default_na=False)
n = sol.targetAllocation.map(lambda s: len(_parse_alloc(s)))
print(f"total solution rows {len(sol):,}")
print(f"  single-key {int((n == 1).sum()):,} ({(n == 1).mean() * 100:.2f}%)")
print(f"  multi-key  {int((n >= 2).sum()):,} ({(n >= 2).mean() * 100:.2f}%)")
print(f"  blank      {int((n == 0).sum()):,} ({(n == 0).mean() * 100:.2f}%)")
