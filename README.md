# Ledgerline

Cash reconciliation matching, evaluated on BenchRec.

**On BenchRec eval — real labelled data from the ICAIF 2023 benchmark — 90.224% of
transactions are auto-closed at 98.368% accuracy.** The remaining 3,133 are escalated to a
human holding 83.61B USD of exposure, ranked so the largest is reviewed first. The batch of
32,048 records is processed in 53.91 s, or 594.4 records per second.

Closed blind — every proposed answer posted without review, which is the only thing
comparable to a plain prediction file — it is **91.7936% match at 95.4665% precision**.

Alongside the dataset in this folder is `MatcherByChatGPT_submission.csv`. Scored with the same
scorer, on the same question — post every answer, abstain where you have none:

| every answer posted blind | match rate | precision | abstention |
|---|---|---|---|
| MatcherByChatGPT_submission.csv | 62.4501% | 95.2503% | 34.4358% |
| this system | 91.7936% | 95.4665% | 3.8474% |

The controller answers a different question, so its figures are not comparable to the table
above and are kept separate. **98.368% is accuracy on the subset it chose to close, not on the
file**; the 95.4665% row is the one to compare against 95.2503%.

| as a controller | |
|---|---|
| auto-closed without review | 28,915 of 32,048 — 90.224% |
| correct among those | 98.368% |
| escalated to a human | 3,133 — 9.776% |

**That prediction file was found next to the dataset. It is not an official baseline, not a
published result, and carries no provenance.** It is here because it is the only other answer
set in the folder, and because it makes a useful point: its 95.2503% precision is bought by
abstaining on 34.4358% of the file, and it scores 0.0000% on every multi-key label (884
predicted, 0 correct). Treat it as a reference point, not as the state of the art.

---

## Why there is no subset-sum solver here

One-to-many reconciliation is commonly formulated as subset-sum: find the ledger rows whose
amounts add to the statement amount. That formulation is well established, and reasonably so.

- A J.P. Morgan paper from August 2025 formalises it as the **Subset Sum Matching Problem**,
and says of it, **in the introduction** rather than the
  abstract: *"it is a critical part of an accounting process known as reconciliation, where
  two sets of financial records are compared to ensure numerical accuracy and agreement."* Two
  subsets match when the absolute difference of their sums is within a tolerance.
  ([arXiv:2508.19218](https://arxiv.org/abs/2508.19218), J.P. Morgan Quantitative Research and
  J.P. Morgan AI Research)
- **Oracle Account Reconciliation** implements it. Its worked example of a 1-to-Many rule
  requires that the *"Sum of Amounts in the subsystem should be equal to the source system
  Amount"*.
  ([Understanding the Transaction Matching Engine](https://docs.oracle.com/en/cloud/saas/account-reconcile-cloud/adarc/admin_trans_match_overview_matching_engine_100x0f827b25.html))
- There are **patents** on it going back over a decade. US8548971B2, *Financial transaction
  reconciliation* (Bank of America, filed 2012, granted 1 October 2013), performs *"subset sum
  comparisons ... between positive subsets of the determined positive subset size and negative
  subsets of the determined negative subset size."*
  ([US8548971B2](https://patents.google.com/patent/US8548971B2/en))
- **Commercial invoice-matching tools** sell it as their core decision logic — ReconcileIQ
  describes *"Subset-sum search across open invoices for that contact. The algorithm tries
  combinations."*
  ([ReconcileIQ](https://bankreconciler.app/blogInvoicePaymentMatching))

We took that formulation on faith. A bounded subset-sum solver over the top-5 candidates was
specified — the enumeration, the tolerance, the tie-breaking when several subsets hit the
target — and was days of work from being written. Nothing had been measured yet. Then we
measured, and the measurement is the only reason it was never built.

### The regimes

Every multi-key group — one statement row answered by several ledger keys — falls into one of
three regimes. **Partition**: the ledger amounts sum to the statement amount. **Repeat**: at
least one ledger row already carries the statement's exact amount. **Neither**.

| regime | train (`matchId` groups) | eval (reconstructed) |
|---|---|---|
| repeat | 83.18% | 79.09% |
| neither | 14.17% | 17.82% |
| **partition** | **2.65%** | **3.09%** |

*6,678 multi-key rows on train, 1,779 on eval.* Train groups come from the `matchId` column.
Eval has no `matchId` — it is blank on all 69,171 rows — so a group's ledger rows are
reconstructed from the keys its label names. Validated against train's own `matchId` groups,
that reconstruction never loses a row, over-collects on 35.58% of groups, and leaves the regime
label unchanged on **99.22%** of rows. The disagreements lean toward reading a partition as a
repeat, so eval's repeat share is if anything a slight over-estimate. Measured by
`findings.py`; see `findings.log`.

**On this data the partition regime describes 2.65% of multi-key cases on train and 3.09% on
eval.** The dominant case is repeat, where several ledger rows already carry the statement's
exact amount.

### What that looks like

Eval `B_id 43581112882`, five ledger rows against a statement amount of 840,311.46:

```
statement rows                          ledger rows
  43581112882   DR   840311.46            551786750288   CR   840311.46   K2
 840304890183   DR   840311.46            594577151758   CR   840311.46   K1
 539698351214   DR   840311.46            173495543687   CR   840311.46   K3
 825752293088   DR   840311.46            244189094128   CR   840311.46   K4
 839660346265   DR   840311.46            517040781446   CR   840311.46   K5

statement amount                840,311.46
sum of all ledger amounts     4,201,557.30   a partition would equal the statement
ledger rows at that amount           5 of 5   a repeat means all of them
```

The target set is all five keys. A solver looking for the subset that sums to 840,311.46 has
nothing to find: the sum is 4,201,557.30, and every single-row subset is an equally good
answer.

### The consequence

When every candidate carries the same amount, arithmetic gives no signal at all. So the
architecture is **retrieval and ranking over candidates, not an arithmetic solver** — the
discriminating evidence is textual, in the reference and attribute fields, and the job of the
model is to rank. Amount blocking still earns its place, but as a filter that pulls the group
together rather than as the thing that distinguishes its members: it cuts the pool from 6,380.9
candidates per query to a mean of 14.08, and then stops helping.

### The boundary condition

Partition is a real regime, and this is not a claim that subset-sum is useless. Razorpay's own
settlement documentation describes exactly it: when the live balance is short, *"we will only
choose the ones that add up to your current live balance"*, and the remaining transactions roll
into the next settlement cycle — a bank credit corresponding to a subset of payments that fit
the available balance, with the remainder deferred.
([About Settlements](https://razorpay.com/docs/payments/settlements/))

The claim is narrower: which regime you are in is an empirical question, it should be measured
before the algorithm is chosen, and in this benchmark the answer was not the one we assumed.

---

## What the system does

A bank statement line arrives. Something in the ledger should account for it — sometimes one
entry, sometimes several, sometimes nothing at all. The task is to name the allocation keys
that answer for it.

This is not a matcher. A matcher answers *what does this pair with*. This answers a second
question: *do I trust that answer enough to post it without a human?* Everything it cannot
defend is escalated with a reason attached, and the escalations are ordered by how much money
is standing behind them, so a reviewer with limited hours spends them on the largest exposures
first.

### Pipeline

**Retrieval** (`retrieve.py`) — TF-IDF over character 3–5 grams. The query is the B-side
reference fields concatenated; the candidates are the A-side fields plus `A_allocation`.
Candidates are blocked by currency and account, then by a date window, then by amount at a
0.01 tolerance. Blocking cuts the pool from 6,380.9 candidates per query (date window only)
to a mean of 14.08, median 1. Top-1 by cosine gives 95.3688% match at 98.4138% precision on
single-key labels.

**Set completion** (`complete.py`) — a single top-1 cannot answer a multi-key label. A
gradient-boosted classifier over 15 features decides, for each of the top-5 candidates,
whether it belongs in the answer set. Trained on BenchRec train only, threshold 0.5, no
eval-label tuning. Overall match 89.9900% → 91.7936%.

**Decision layer** (`controller.py`) — auto-close or escalate. Four triggers, each derived
from a measurement rather than a guess, and each scored on whether it catches more wrong
answers than right ones. Escalations are labelled with one of five exception classes. Every
decision is written to a JSONL audit trail with the candidates considered, their scores and
amount deltas, the triggers that fired, and the answer proposed.

**Investigation** (`investigate.py`) — an LLM writes a plain-English explanation of why a row
was escalated. It never makes, changes or ranks a match, and it is given no labels and no gold
answer. Every number it writes is checked against the evidence it was passed; ungrounded
output is surfaced, not hidden.

`exposure.py` re-weights the decisions already made by money. `export.py` serialises them to
static JSON. `web/` is a review interface over that JSON. `check.py` is a Playwright layout
check over the interface.

---

## Why this matters to a payment company

Nothing below was measured by this project — it is context, and every claim is a link. This
system reconciles a benchmark dataset, not anyone's production ledger.

**Reconciliation is a deadline, not a background job.** Under the RBI's *Guidelines on
Regulation of Payment Aggregators and Payment Gateways*, clause 8.6: *"At the end of the day,
the amount in escrow account shall not be less than the amount already collected from customer
as per 'Tp' or the amount due to the merchant."* The same guidelines set a reporting calendar —
*Statistics of Transactions Handled* monthly by the 7th, and an *Auditors' Certificate on
Maintenance of Balance in Escrow Account* quarterly by the 15th of the following month.
([RBI, 17 March 2020](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11822))
That is the condition the exposure-ranked queue is built for: a fixed number of hours before a
deadline, and a decision about which items to spend them on.

**Unresolved items have a published per-day price.** RBI's *Harmonisation of Turn Around Time
(TAT) and customer compensation for failed transactions* sets, for a failed ATM withdrawal,
*"₹ 100/- per day of delay beyond T + 5 days, to the credit of the account holder"*, with
comparable per-day amounts across card, UPI, IMPS and wallet failures.
([RBI/2019-20/67, 20 September 2019](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11693))
An ageing exception is not a neutral backlog item.

**One-to-many settlement is a documented product structure.** Razorpay Route *"splits payments
into various portions for seamless transfer to multiple parties"*, and its API exposes
[Fetch Transfers for a Settlement](https://razorpay.com/docs/api/payments/route/fetch-transfers-for-a-settlement/)
— all the transfers made for one `recipient_settlement_id`. One credit, many underlying
transfers, which is the shape the multi-key label describes.
([Route](https://razorpay.com/docs/payments/route/))

**Near-duplicate records have a real mechanism behind them.** Razorpay's webhook documentation
states plainly that *"You could be receiving the same events multiple times as Razorpay follows
at-least-once delivery semantics"* and that *"you may not always receive the webhooks in
order"*, recommending an idempotency check on `x-razorpay-event-id`.
([Webhook best practices](https://razorpay.com/docs/webhooks/best-practices/))
That is the class this system handles worst: `duplicate_reference` is where set completion
falls apart out of domain, losing 50.546 points (finding 5).

**What this does not do.** It reconciles a static batch at line level, matching statement rows
to ledger allocation keys. It does not reconcile balances, and it has no notion of time passing
after the batch: a reversal, a chargeback or a refund landing weeks later is not something it
models. Both are ordinary requirements in production reconciliation and neither is addressed
here.

---

## Datasets, and what each is for

| file | rows | role |
|---|---|---|
| `BenchRec_cash_v1.0_train.csv` | 68,975 B rows | fits the completion classifier and every threshold |
| `BenchRec_cash_v1.0_eval.csv` | 32,048 B rows | produces every headline number |
| `BenchRec_cash_v1.0_solution.csv` | 32,048 | eval labels, used for measurement only |
| `MatcherByChatGPT_submission.csv` | 32,048 | third-party prediction file, comparison only |
| `synth_transactions.csv` | 158,534 (A 108,534, B 50,000) | per-class breakdowns, throughput, domain shift |
| `synth_small_transactions.csv` | 160 (A 110, B 50) | the stated 50+ record requirement |

**Train** fits models and thresholds. The completion classifier is fitted on 19,339 candidate
decisions from train, 8,905 positive (46.05%). The `low_confidence` grid was swept against
train's inline `targetAllocation` labels; eval and synthetic were not consulted.

**Eval** produces every headline number. Its label mix: 30,057 single-key (93.79%), 1,779
multi-key (5.55%), 212 blank (0.66%).

**Synthetic** satisfies the stated 50+ record minimum, and does three jobs train and eval
cannot: it gives per-class breakdowns (the real data has no class labels), it gives clean
throughput figures at scale, and it is the domain-shift test — the completion classifier is
applied to it frozen, without retraining, refitting or recalibration.

Both synthetic batches pass a difficulty gate set in advance: if exact amount matching scored
above 90%, the data would be too easy to be worth keeping. Measured 56.000% on the 50-group
batch and 57.000% on the 50,000-group batch.

**No headline number comes from data we generated.** 90.224%, 98.368%, 3,133, 83.61B and
594.4 rec/s are all BenchRec eval. Synthetic numbers are labelled as such wherever they
appear.

The split is temporal, which matters: train runs 2015-03-08 to 2023-03-05, eval 2022-12-21 to
2023-05-31, and only 0.8238% of eval rows fall at or before train's last date. Allocation keys
embed a date, so a key seen in train can essentially never be reused verbatim in eval.

---

## What broke

### 1. Multi-key groups repeat rather than partition

**Assumed.** A multi-key label means several ledger entries that sum to the statement amount,
which makes this a subset-sum problem. A bounded solver over the top-5 was scoped.

**Measured.** Partition is 2.65% of multi-key groups on train and 3.09% on eval. The dominant
regime is repeat, at 83.18% and 79.09%, where the ledger rows already carry the statement's
exact amount and arithmetic cannot separate them.

**Changed.** The solver was dropped before being written, in favour of a classifier over
candidate ranks. Worth **+1.8036 points** of overall match rate.

**In full:** [Why there is no subset-sum solver here](#why-there-is-no-subset-sum-solver-here)
— the citations for the formulation, the regime table for both splits, the validated eval
reconstruction, the worked example, and the boundary condition.

### 2. Float precision at 6.6 billion

**Assumed.** An amount tolerance of 0.01 means "within one cent", and `abs(a - b) <= 0.01`
expresses it.

**Measured** (`findings.log`). The largest absolute amount in eval is 6,567,592,109.00.
float64 resolution at that magnitude is 9.5367431640625e-07. Two values exactly one cent
apart subtract to `0.009999997913837433` — under the threshold, but only just, and by an
accident of representation rather than by design. The comparison was not reliably expressing
what it claimed to express.

**Changed.** Every amount is carried and compared as an integer number of cents. The
one-cent difference above is exactly `1`.

**Consequence.** The tolerance is now exact rather than approximate. From the tolerance sweep
in `retrieve.log`, the reachable ceiling at exact-amount blocking is 96.0808% and at a
0.01 tolerance is **96.7196%**. Which of those two the comparison was actually implementing
had been left to floating-point noise at the boundary.

### 3. The digit-run signal is real, redundant as a boost, harmful as a filter

**Assumed.** Bank references and ledger references share long digit runs. Matching on a shared
run of 7–12 digits should be a strong signal worth adding.

**Measured** (`retrieve.log`). The signal is real. The denominator is single-key labels only —
the log reads *"True matches tested (single-key labels): 30,057"* — and of those 30,057, 19,394
share a digit run of length 7–12, **64.5241%**. But after amount blocking has already cut the
pool to ~15 candidates, 42.7661% of the *surviving* candidates share a run with the query,
which is a weak discriminator. And the redundancy check settles it:

```
queries where the boost COULD reorder (some but not all candidates share): 6,131
of those, cosine's top-1 ALREADY shares a run:                             6,131  (100.0000%)
```

**Changed.** Neither form shipped. As a hard filter it collapses match rate by 32.5615 points
(62.8073%) because 34.10% of queries have no surviving candidate sharing a run at all,
forcing an abstention on every one. As a boost it changes nothing.

**Consequence.** As a boost, `+0.0000 pts` on every metric — top-1, not-in-top-5, match rate,
precision. The character n-grams *are* scoring on the shared run; cosine had already absorbed
it. As a filter it is not neutral but actively harmful on match rate, though it does buy
precision: 99.4521%, **+1.0383 points**, for 32.5615 points of match rate. That is the same bad
exchange rate as finding 6, and it was refused for the same reason.

### 4. Dropping `B_transactionReferences` cost 2.33 points

**Assumed.** Since the digit-run signal turned out to be redundant, and the reference field is
where the digit runs live, that field was assumed to be carrying nothing the attributes field
did not already carry. Experiment A dropped it and built the query from
`B_transactionAttributes` alone.

**Measured** (`retrieve.log`):

| | top-1 | precision | not in top-5 |
|---|---|---|---|
| cosine + amount (reference) | 95.3688% | 98.4138% | 3.5499% |
| attributes only | 93.0399% | 96.0106% | 4.2752% |
| | **−2.3289 pts** | **−2.4032 pts** | +0.7253 pts |

**Changed.** Nothing — the reference field stayed.

**Consequence.** "Carries no digit-run signal" was not the same claim as "carries no signal",
and conflating them would have cost 2.3289 points. Finding 3 and finding 4 are about the same
field and point in opposite directions; both were measured separately rather than inferred
from one another.

### 5. Completion is +1.80 on real data and −2.30 on synthetic

**Assumed.** A component that helps on eval helps in general.

**Measured.** On BenchRec eval, completion at threshold 0.5 moves overall match
89.9900% → 91.7936% (**+1.8036**) and overall precision 93.5908% → 95.4665%. It gains 800
multi-key rows and loses 222 single-key rows, since a wrongly added key breaks an otherwise
correct answer. On the synthetic 50,000-group batch, applied frozen, the same component moves
overall match 84.330% → 82.030% (**−2.300**).

The per-class table locates it exactly (`synth_run.log`):

| class | rows | match, completion off | match, completion on | delta |
|---|---|---|---|---|
| duplicate_reference | 4,024 | 75.944% | 25.398% | **−50.546** |
| one_to_one | 46,944 | 89.328% | 84.507% | −4.821 |
| repeat | 1,855 | 0.000% | 60.000% | +60.000 |

The gain does not offset the loss because the classes are not the same size: `repeat` is 1,855
rows and `duplicate_reference` is 4,024. Sixty points on the smaller class loses to fifty on
one more than twice as large.

**Changed.** Nothing was retuned to make the synthetic number look better — that would have
destroyed the test. The result is reported as the generalisation signal it is.

**Consequence.** Completion does exactly what it was built to do (repeat groups go from 0% to
60%), and it does it while being badly miscalibrated on duplicated references it never saw in
training. On out-of-domain data the losses exceed the gains.

### 6. The `low_confidence` trigger, dropped

**Assumed.** A low top-1 similarity, or a thin margin between rank 1 and rank 2, indicates an
answer not worth trusting.

**Criterion, set before looking at results.** Keep the trigger only if the rows it escalates
would have been *wrong* more often than right — a trigger whose escalations are mostly correct
answers is buying precision by giving up coverage indiscriminately. Threshold: fewer than 50%
of escalated rows already correct.

**Measured** (`ctrl.log`). 42 grid points swept on train only — `min_top1` in
{0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15} × `min_margin` in
{0.000, 0.005, 0.010, 0.020, 0.040, 0.080}. One point disables the trigger, so 41 fired. **The
criterion failed at all 41.** The best available was `min_top1=0.00, min_margin=0.010` at
86.568% already-correct — still escalating 6.4 correct rows for every wrong one.

**Changed.** Dropped, by setting both thresholds to 0.0. It never fires on eval or synthetic.

**Consequence.** It is worth being precise about what was given up, because the trigger was
not doing nothing. At that same best point, auto-close precision on train rises from 96.056%
to 98.870% — nearly three points of real precision. But it costs coverage: auto-close falls
from 87.332% to 58.499% of the file. That is 28.8 points of coverage for 2.8 points of
precision, most of it spent escalating answers that were already right. The trigger was buying
something real at a bad exchange rate, which is harder to notice than a trigger that does
nothing at all.

### 7. The LLM transposed a figure inside an otherwise correct paragraph

**Assumed.** An explanation layer that is given the evidence and told to use only the evidence
will use only the evidence.

**Measured.** Of 58 explanations generated, 57 pass the automated grounding check and 1 does
not. `b_id 745373354167` (synthetic batch, `fee_band_match`,
`models/gemini-3.5-flash-lite`). The evidence gave five candidate deltas: 3.96, 4.82,
**1.49**, 12.29, 8.99. The model wrote:

> ...yielding respective deltas from B of 3.96, 4.82, **1.29**, 12.29, and 8.99.

Every other figure in the paragraph is correct: the transaction amount, its cents form, all
five candidate amounts, the other four deltas, the pool size, the exception class. One digit
pair transposed, buried in an otherwise accurate sentence — exactly the error a human reviewer
skims past.

**Changed.** Nothing about the prompt. The grounding check already caught it: every numeric
token in the output is matched against the numbers in the evidence, and `1.29` had no source.

**Consequence.** The check is the deliverable, not the explanation. Ungrounded output is
surfaced in the interface with the claimed figure next to what was actually in the evidence,
rather than being suppressed or silently regenerated. A separate check confirms no response
proposed a match: 58 of 58 clean.

---

## Evaluation

### Two precision measures

| | by rows | by value |
|---|---|---|
| auto-close precision | 98.368% | 98.815% |
| auto-closed | 28,915 (90.22%) | 1,166.76B (93.31%) |
| escalated | 3,133 (9.78%) | 83.61B (6.69%) |

Value-weighted precision is **0.447 points better** than row-count precision. The reason is in
the means: a wrongly auto-closed row is worth 29.30M on average, a correctly auto-closed row
40.53M. Errors fall disproportionately on smaller-than-average transactions, which is the
favourable direction and is invisible if you only count rows.

Of the 1,250.37B total in the batch: 1,152.93B auto-closed correctly (92.207%), 83.61B in the
exception queue (6.687%), 13.83B auto-closed incorrectly across 472 rows (1.106%).

Auto-close precision is flat across magnitude — the trend across the 10 buckets with at least
30 auto-closed rows is +0.063 points per bucket, and the $1e9–1e10 bucket is 100.000% over 156
rows. The system is neither better nor worse on large amounts.

### The exposure-retired curve

What share of queue exposure is retired by reviewing the top N%, ranked by exposure against a
random order (mean of 20 seeds):

| review top | rows | ranked | random | sd | gain |
|---|---|---|---|---|---|
| 1% | 31 | 45.408% | 0.655% | 0.561 | +44.753 |
| 5% | 157 | 77.824% | 4.759% | 2.601 | +73.065 |
| **10%** | **313** | **87.529%** | **10.036%** | 3.513 | **+77.493** |
| 25% | 783 | 98.306% | 23.144% | 3.556 | +75.162 |
| 50% | 1,566 | 99.735% | 46.447% | 5.329 | +53.288 |

Reviewing 313 of 3,133 escalations retires 87.529% of the money at risk. Random order tracks
the diagonal, which is the point of showing it — the ranking is worth having only in the gap
between the two columns.

### Per-trigger verdicts, BenchRec eval

| trigger | escalated | would be correct | verdict |
|---|---|---|---|
| `no_candidate` | 1,408 | 12.429% | earning its keep |
| `completion_added` | 1,725 | 46.377% | mixed |
| `fee_band_only` | 0 | — | never fired |
| `low_confidence` | 0 | — | dropped, see finding 6 |

By exception class: `missing_counterparty` 1,408 rows / 49.85B / 12.429% already correct;
`incomplete_set` 1,334 / 26.16B / 55.997%; `duplicate_reference_suspected` 391 / 7.60B /
13.555%.

Fee-band widening is configured per batch, and both settings are run against both batches so
the choice is visible. It is off for eval — real BenchRec has effectively no fee deductions,
so widening admits only false candidates (overall match 91.794% → 75.181%) and additionally
masks the `no_candidate` trigger by guaranteeing a non-empty pool (1,408 firings → 165). It is
on for synthetic, which contains fees by construction, where it gains +4.750 points.

---

## Limitations

**The synthetic result is the real generalisation signal, and it is worse.** Auto-close
coverage is 78.770% at 98.431% precision against 90.224% at 98.368% on eval, and completion
turns from +1.80 to −2.30. On synthetic, value-weighted precision is 0.224 points *worse* than
row-count precision — errors there fall on larger-than-average rows, the opposite of eval.

**The synthetic data is ours.** It inherits its generator's assumptions, and the `neither` and
`partition` classes score 0.000% under every configuration tried.

**Investigations cover 58 of a targeted 100**, of which 30 are eval rows and appear in the
exported detail files. The run stopped on free-tier quota exhaustion, not because it finished.

**The amount tolerance discards true matches.** At 0.01, 3.2804% of true matches are outside
the block and can never be recovered by any similarity function. Widening to 1.00 recovers
some (ceiling 97.6345%) but costs precision (97.9746%); widening to 1% relative reaches a
98.4163% ceiling but drops precision to 89.5623%. The setting is a deliberate trade, not a
free one.

**`incomplete_set` escalates mostly-correct rows.** 55.997% of its 1,334 escalations would
have been right if closed blind. It survives the same criterion that killed `low_confidence`
only because it is close to the line, not comfortably past it — and unlike `low_confidence`
its 44% wrong share concentrates 24.20B of caught exposure.

**The interface shows one batch at a time and holds the whole queue in memory.** The queue is
fetched in pages of 1,000 rows so no single request is large (the synthetic index is 211,923
bytes against 2,209,247 unpaged), but every page is kept, and the treemap is not drawn until
all of them have landed.

Eval labels were used for measurement only. No threshold, feature or hyperparameter was
selected against them.

---

## Reproduce

Python 3.12 with `pandas`, `numpy`, `scikit-learn`. Run from the project directory. The
BenchRec CSVs are not in this repository — obtain them from the ICAIF 2023 benchmark and place
them alongside the scripts.

```bash
# 1. Understand the data — inventory, schema, label completeness, allocation structure
python explore_benchrec.py            # writes benchrec_report.md

# 2. Score the third-party submission, and test whether the split is temporal
python score.py                       # 62.4501% match / 95.2503% precision

# 3. Retrieval, both experiments, the amount-tolerance sweep
python retrieve.py > retrieve.log     # ~1 min

# 4. Set completion: measurement, recall ceiling, classifier, threshold sweep
python complete.py > complete.log     # ~3 min

# 5. Generate synthetic data at both scales, validate the difficulty gate
python generate.py > gen.log          # ~4 min

# 6. Full pipeline over both synthetic batches (out-of-domain test)
python run_synth.py > synth_run.log   # ~8 min

# 7. Decision layer: threshold fit on train, then eval + synthetic
python controller.py > ctrl.log       # ~9 min

# 8. Value-weighted measurement over the decisions from step 7
python exposure.py > exposure.log     # ~1 min

# 9. Signed balance drift over the same decisions — direction and cancellation
python drift.py > drift.log           # ~1 min

# supporting measurements for the findings above
python findings.py > findings.log     # ~1 min
```

Steps 3–7 are independent except that step 7 reads the synthetic files from step 5, and steps 8
and 9 read the audit files from step 7.

### The explanation layer and the interface

```bash
cp .env.example .env                  # then add ONE key; .env is gitignored
python investigate.py --rank-top 100 --rpm 10     # free tier is ~10 req/min; backs off on 429
python export.py                                  # static JSON -> web/data/  (--synth for both)
cd web && python -m http.server 8000              # then open http://localhost:8000/
```

`export.py` recomputes the summary and curve figures from the audit files directly — it never
parses `exposure.log` — and asserts every figure against `exposure.py`'s own implementation
before writing, failing loudly on any disagreement.

### Checkers

```bash
python check.py                       # Playwright layout check at 1440x900, 1280x800, 1920x1080
python check.py --width 1366          # one width
```

`check.py` serves `web/`, screenshots the cover, every section and the detail panel at each
width into `screenshots/`, and asserts no horizontal overflow, nothing past the viewport edge,
no text clipped by its own box, no truncated nav labels, and every queue amount fully visible
above its magnitude bar. Failures name the element and the number that broke.

### Outputs

`benchrec_report.md`, `retrieve_predictions*.csv`, `complete_predictions.csv`,
`synth_transactions.csv` / `synth_solution.csv` / `synth_manifest.json` (plus `synth_small_*`),
`controller_audit_eval.jsonl` (32,048 records), `controller_audit_synth.jsonl` (50,000),
`controller_exceptions_eval.csv` (3,133), `controller_exceptions_synth.csv` (10,615),
`exceptions_ranked_eval.csv` (3,133 rows, 83.61B exposure), `exceptions_ranked_synth.csv`
(10,615 rows, 633.76B), `investigations.jsonl`, and `web/data/` — 13,767 files totalling
24.45 MB with `--synth`, or the eval batch alone without it.

`web/data/` is gitignored — 13,767 files is not something to put in a repository, and it
regenerates from the audit files in one command. So is the source data and anything derived
from it row by row; all of it comes back from the commands above. The audit JSONL carries the candidates considered, their scores and amount
deltas, which triggers fired, the decision and the evidence, so any single decision can be
reconstructed.

Every number in this README appears in `score.log`, `retrieve.log`, `complete.log`, `gen.log`,
`synth_run.log`, `ctrl.log`, `exposure.log`, `drift.log`, `export.log`, `investigate.log`,
`findings.log`
or `investigations.jsonl`, with three exceptions, all deliberate. A figure written as a
difference (−2.300, −50.546, −4.821, and the coverage-for-precision trade in finding 6) is the
subtraction of two logged numbers, and both operands are printed beside it. File sizes and
counts (211,923 and 2,209,247 bytes) are measured off disk. And the identifiers, dates and
figures in the two cited sections — *Why there is no subset-sum solver here* and *Why this
matters to a payment company* — belong to external sources, not to this project: arXiv:2508.19218,
US8548971B2, RBI Id=11822 and Id=11693, the ₹100/- per day. Each is linked, and each quotation
was checked character by character against the source page rather than paraphrased from memory.
