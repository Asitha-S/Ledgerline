# Ledgerline

Cash reconciliation matching, evaluated on BenchRec.

**On BenchRec eval, 90.224% of transactions are auto-closed at 98.368% accuracy.**

That is 28,915 of 32,048 transactions closed without review, 3,133 escalated for a human.
Throughput 621.8 records/sec. If every row were closed blind instead, overall match rate
would be 91.794%. (`ctrl.log`, FINAL REPORT — BenchRec eval)

These numbers come from BenchRec_cash_v1.0, real labelled cash reconciliation data from the
ICAIF 2023 benchmark competition. They do not come from data we generated.

### Comparison

Alongside the dataset in this folder is `MatcherByChatGPT_submission.csv`. Scored with the
same scorer:

| | match rate | precision | abstention |
|---|---|---|---|
| MatcherByChatGPT_submission.csv | 62.4501% | 95.2503% | 34.4358% |
| this system (auto-closed rows) | — | **98.368%** | 9.776% escalated |
| this system (all rows closed blind) | **91.794%** | 95.466% | — |

**That submission is a prediction file found alongside the dataset. It is not an official
published baseline, and we do not know who produced it or under what conditions.** Treat it
as a reference point, not as the state of the art. Its 95.2503% precision is bought with a
34.4358% abstention rate; it also scores 0.0000% on all 1,779 multi-key labels, because it
never emits more than one key (`score.py` output).

---

## What the system does

Given a bank statement entry (a "type B" transaction), find which internal ledger entries
(type A) it reconciles against, and emit the **allocation key** of those entries — a string
of the form `CURRENCY_DATE_ACCOUNT_ATTRIBUTES`. The answer can be one key, several keys, or
none at all when the transaction genuinely has no counterpart.

Then decide, per transaction, whether the answer is good enough to post automatically or
should go to a human with a reason attached.

## Architecture

Three layers, each independently scored.

**1. Retrieval — `retrieve.py`**
Blocks candidates by currency, account, a ±7-day value-date window, and an amount match in
integer cents. Scores the survivors with TF-IDF over character 3–5-grams, cosine similarity,
top-5, keeps top-1. On eval single-key labels: 95.3688% match, 98.4138% precision
(`retrieve.log`).

**2. Completion — `complete.py`**
Retrieval emits exactly one key, so every multi-key label scores zero. A gradient-boosted
classifier decides, for each of the remaining top-5 candidates, whether it belongs in the
set. Trained on BenchRec train only: 19,339 candidate decisions, 8,905 positive (46.0468%).
On eval it lifts overall match 89.9900% → 91.7936% (`complete.log`).

**3. Decision layer — `controller.py`**
Auto-close or escalate. Four triggers, each traceable to a measurement: `completion_added`,
`fee_band_only`, `low_confidence` (dropped — see findings), `no_candidate`. Every escalated
row gets one of five exception classes. Every transaction gets a JSONL audit record with
candidates, scores, triggers, decision and evidence.

`score.py` is the scorer used everywhere: exact set equality, blank labels scored not
dropped, missing rows counted as abstentions.

---

## Datasets

| file | rows | role |
|---|---|---|
| `BenchRec_cash_v1.0_train.csv` | 149,854 (A 80,879, B 68,975) | fits the completion classifier and the decision thresholds |
| `BenchRec_cash_v1.0_eval.csv` | 69,171 (A 37,123, B 32,048) | **produces every headline number** |
| `BenchRec_cash_v1.0_solution.csv` | 32,048 | eval labels |
| `MatcherByChatGPT_submission.csv` | 32,048 | third-party prediction file, comparison only |
| `synth_transactions.csv` | 158,534 (A 108,534, B 50,000) | per-class breakdowns, throughput, domain-shift test |
| `synth_small_transactions.csv` | 160 (A 110, B 50) | 50 groups / 50 B records — satisfies the stated 50+ record requirement |

Eval label mix: 30,057 single-key (93.79%), 1,779 multi-key (5.55%), 212 blank (0.66%)
(`findings.log`).

**No headline number comes from generated data.** The synthetic batches exist to answer
questions the real data cannot: per-corruption-class accuracy (real data has no corruption
labels), throughput at a chosen scale, and whether the classifier survives domain shift.
Synthetic results are reported separately and always labelled as such.

The generator (`generate.py`) is config-driven and gated: if exact amount matching scores
above 90% the data is too easy to ship. It scores 57.000%, 33 points below the gate
(`gen.log`).

---

## Findings

### 1. Multi-key groups repeat rather than partition

The instinct is that a multi-key allocation splits the transaction — several ledger entries
summing to the statement amount. It does not. In 83.18% of multi-key train rows the amounts
**repeat** B's amount; only 2.65% partition it; 14.17% do neither. In 70.23% of multi-key
rows *every single* A row in the group carries B's exact amount (`findings.log`).

`matchId 184541000741` — 6 A rows, 6 B rows, 6 distinct allocation keys:

```
B rows (external statement)          A rows (internal ledger)
160725871107  DR  203,235.00         883399681180  CR  203,235.00   K1
670730959472  DR  203,235.00         434384224841  CR  203,235.00   K2
996827316754  DR  203,235.00         680184429448  CR  203,235.00   K3
293269702131  DR  203,235.00         444196382816  CR  203,235.00   K4
123784971159  DR  203,235.00         258049165276  CR  203,235.00   K5
462933939786  DR  203,235.00         991589274456  CR  203,235.00   K6

B amount                    203,235.00
sum of all A amounts      1,219,410.00    (a partition would equal B)
A rows with amount == B          6 of 6   (a repeat means all of them)
target set        [K1, K2, K3, K4, K5, K6]
```

Six identical amounts on each side, six different keys, sum 6× B. Consequence: amount is
useless for choosing *among* members of a multi-key group, because it is identical across
all of them. It is what pulls them into the candidate pool together. Subset-sum would have
been the wrong tool.

### 2. Float precision at 6.6e9, and the switch to integer cents

The largest absolute amount in eval is 6,567,592,109.00. At that magnitude a float64 has a
resolution of 9.5367431640625e-07. Two values exactly one cent apart subtract to
0.009999997913837433 — so a `<= 0.01` tolerance is not reliably representable
(`findings.log`).

All amount comparisons are done in integer cents. This is not cosmetic: it moved the
measured true-match ceiling for the 0.01 tolerance from 96.3968% to 96.7196%. Roughly 0.32
points of apparent headroom were a floating-point artefact.

### 3. The digit-run negative result

The linking signal between the two sides is a shared ~9-digit reference run embedded in
`B_transactionAttributes` and the A-side text. It is real: 19,394 of 30,057 true matches
(64.5241%) share a digit run of length 7–12 (`retrieve.log`).

Extracting it explicitly changed nothing. As a score boost it altered **zero** predictions.
The boost *could* have reordered 6,131 queries — those where some but not all surviving
candidates share a run. On **6,131 of 6,131 (100.0000%)** cosine already ranked a shared-run
candidate first. The character n-grams were already scoring on the digit run; naming it
explicitly re-adds information the model had absorbed.

As a hard filter it was actively harmful: match rate 95.3688% → 62.8073% (−32.5615) to buy
+1.0383 of precision, because on 34.10% of queries no surviving candidate shares a run at all.

### 4. Dropping the dead field made it worse

`B_transactionReferences` shares a digit run with the A side on only 3.16% of true matches,
against 63.69% for `B_transactionAttributes`. It looked like dead weight diluting the query.

Removing it cost 2.33 points: single-key match 95.3688% → 93.0399%, precision 98.4138% →
96.0106%, not-in-top-5 3.5499% → 4.2752% (`retrieve.log`).

"Carries no digit-run signal" is not "carries no signal". Mean query length falls from 50.3
to 30.7 characters, and the shorter query gives the n-gram cosine less to discriminate with
among near-identical candidates. The field was contributing through a channel we had not
measured.

### 5. Domain shift: completion helps on real data and hurts on synthetic

Same frozen classifier, trained on BenchRec train, never retrained:

| | completion OFF | completion ON | delta |
|---|---|---|---|
| BenchRec eval | 89.9900% | 91.7936% | **+1.8036** |
| synthetic 50k | 84.330% | 82.030% | **−2.300** |

On real eval it gains 800 rows and loses 222, net +578 (`complete.log`). On synthetic it
gains 60 points on the `repeat` class (0.000% → 60.000%) but the overall number goes
backwards, and the cause is one class: **`duplicate_reference` collapses 75.944% → 25.398%,
a loss of 50.5 points** (`synth_run.log`).

Those groups contain a near-clone A row — same amount, same reference string, attributes
plus one word. The classifier reliably accepts it as a second key and breaks answers that
were correct. That corruption does not exist in BenchRec train, so the model was never
taught to reject it. On the `repeat` class the transfer works: of 1,855 groups, 1,113
recovered the full key set exactly, 74 added only correct keys but stopped short, 135 added
at least one wrong key, and 533 added nothing.

### 6. The low_confidence trigger: swept on train, failed, dropped

The trigger escalates a transaction when top-1 similarity or the rank1–rank2 margin falls
below a floor. On an early eval run it escalated 17,024 rows of which 68.7% were already
correct.

Criterion set in advance: keep the setting where the trigger's would-have-been-correct rate
falls **below 50%** — catching more wrong than right — while maximising auto-close coverage.

42 grid points were swept on `BenchRec_cash_v1.0_train.csv` using train's inline labels, and
on nothing else. 41 of them fire the trigger. **All 41 failed.** The best result across the
entire grid is 86.568% already-correct (min_top1=0.0, min_margin=0.01) — not close to the
bar. The marginal view, counting only rows the trigger alone removes from auto-close, is
worse at 90.345% (`ctrl.log`).

The trigger was dropped.

**It was not doing nothing.** On train it lifts auto-close precision from 96.056% to 98.870%
— it buys real precision, at 29 points of coverage. It is a bad exchange rate, not a useless
signal. If auto-close precision matters more than coverage in your workflow, the grid is in
`ctrl.log` and the least-bad firing setting is visible there. We did not ship it because it
fails the criterion we set before looking.

Dropping it moved eval auto-close coverage from 37.350% to 90.224%, with auto-close
precision going 99.148% → 98.368%: 0.78 points of precision for 53 points of coverage.

---

## Limitations

**106 confidently-wrong completions that no threshold separates.** Sweeping the completion
acceptance threshold from 0.50 to 0.95, single-key rows broken by a wrongly added key fall
only from 222 to 106. Half the damage is done by candidates the classifier scores ≥ 0.95. No
threshold in that range restores single-key precision to the 98.4138% completion-off
baseline — the closest is 0.95 at 98.0499%, still 0.35 short, and by then multi-key match
rate has fallen from 44.9691% to 21.5852% (`complete.log`). Fixing this needs a better
feature or a rejection model, not a threshold.

**Fee handling requires per-dataset configuration.** Widening the amount block to admit
candidates up to 5% below B's amount is correct for data containing fees and wrong for data
without them:

| batch | widening | auto coverage | auto precision | overall match | no_candidate fires |
|---|---|---|---|---|---|
| BenchRec eval | OFF | 90.224% | 98.368% | 91.794% | 1,408 |
| BenchRec eval | ON | 74.260% | 98.193% | 75.181% | 165 |
| synthetic 50k | OFF | 84.410% | 93.996% | 82.030% | 3,944 |
| synthetic 50k | ON | 78.770% | 98.431% | 86.780% | 0 |

Real BenchRec has effectively no fee deductions, so widening admits only false candidates and
costs 16.6 points of overall match. It also **silently disables the `no_candidate` trigger**
by guaranteeing a non-empty pool — firings drop 1,408 → 165 on eval and 3,944 → 0 on
synthetic. There is no single setting that is right for both, so it is per-batch config in
`controller.py` (`BATCH_FEE_WIDENING`) and has to be set deliberately per dataset.

**`incomplete_set` escalates 55.997% already-correct rows.** On eval it accounts for 1,334
of 3,133 escalations and more than half of them would have been right if auto-closed
(56.480% on synthetic). By the same criterion that killed `low_confidence`, this trigger is
costing coverage for little return and is a candidate for the same treatment. It was left in
because the fit was scoped to `low_confidence`; re-running the sweep with it included is the
obvious next step. `fee_band_only` on synthetic has the same problem at 59.382%.

**Other limits.** Multi-key match rate is bounded at 48.17% by top-5 retrieval — 19.45% of
multi-key rows have no gold key in the top-5 at all, so completion cannot reach them. The
`neither` and `partition` classes score 0.000% under every configuration tried. Synthetic
results are from data whose generator we wrote, and inherit its assumptions.

---

## Reproduce

Requires Python 3.12 with `pandas`, `numpy`, `scikit-learn`. Run from the project directory.

```bash
# 1. Understand the data — inventory, schema, label completeness, allocation structure
python explore_benchrec.py            # writes benchrec_report.md

# 2. Score the third-party submission
python score.py                       # 62.4501% match / 95.2503% precision

# 3. Retrieval + the two experiments + the amount-tolerance sweep
python retrieve.py > retrieve.log     # ~1 min

# 4. Set completion: measurement, ceiling, classifier, threshold sweep
python complete.py > complete.log     # ~3 min

# 5. Generate synthetic data at both scales and validate the difficulty gate
python generate.py > gen.log          # ~4 min

# 6. Full pipeline over both synthetic batches (out-of-domain test)
python run_synth.py > synth_run.log   # ~8 min

# 7. Decision layer: threshold fit on train, then eval + synthetic
python controller.py > ctrl.log       # ~9 min

# supporting measurements for the findings above
python findings.py > findings.log     # ~1 min
```

Steps 3–7 are independent of each other except that step 7 reads the synthetic files written
by step 5. Every number in this README appears in one of `retrieve.log`, `complete.log`,
`gen.log`, `synth_run.log`, `ctrl.log`, `findings.log`, or the stdout of `score.py`.

### Outputs

`benchrec_report.md`, `retrieve_predictions*.csv`, `complete_predictions.csv`,
`synth_transactions.csv` / `synth_solution.csv` / `synth_manifest.json` (plus `synth_small_*`),
`controller_audit_eval.jsonl` (32,048 records), `controller_audit_synth.jsonl` (50,000
records), `controller_exceptions_eval.csv` (3,133 rows), `controller_exceptions_synth.csv`
(10,615 rows).

The audit JSONL carries candidates considered, their scores and amount deltas, which triggers
fired, the decision and the evidence, so any single decision can be reconstructed. Candidates
are logged by `a_id` rather than by full allocation key; the key is derivable from the
transactions file.
