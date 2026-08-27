"""
score.py — scoring for the BenchRec cash reconciliation task.

Public API:

    score(predictions_df, solution_path) -> dict

Everything else is private (leading underscore). No matcher, no modelling.

Running this file directly scores MatcherByChatGPT_submission.csv against
BenchRec_cash_v1.0_solution.csv and prints a temporal-split check on B_valueDate.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ----------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------
def _parse_alloc(value) -> set:
    """Parse a targetAllocation cell into a set of allocation keys.

    Two encodings are in play, and they are NOT the same across files:

      * solution file    bare key            USD_2023-03-05_ACC#00001_669...
                         bracketed list      [USD_..._KEY1,USD_..._KEY2]        (unquoted)
      * submission file  JSON-ish list       ["USD_..._KEY1"]                   (quoted)
                         empty list          []

    An empty string, an empty list, and a missing value all parse to the empty set.

    Whitespace is significant INSIDE an allocation key (the attribute field is
    space-padded), so element text is never whitespace-stripped. The only trimming
    done is removal of a matched pair of surrounding quotes, which is an artefact of
    the JSON-ish encoding rather than part of the key.

    Commas do not appear inside individual keys, so splitting on ',' is safe.
    """
    if value is None:
        return set()

    # pandas may hand us NaN (a float) for missing cells.
    if isinstance(value, float) and pd.isna(value):
        return set()

    s = str(value)
    if s.strip() == "" or s.strip().lower() == "nan":
        return set()

    stripped = s.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        if inner.strip() == "":
            return set()
        parts = inner.split(",")
    else:
        parts = [s]

    keys = set()
    for part in parts:
        key = _unquote(part)
        if key != "":
            keys.add(key)
    return keys


def _unquote(part: str) -> str:
    """Remove one matched pair of surrounding quotes, if present.

    Only strips whitespace when doing so reveals a quoted element — otherwise the
    element is returned verbatim, because leading/trailing spaces are part of the key
    in the bare encoding.
    """
    candidate = part.strip()
    if len(candidate) >= 2 and candidate[0] in "\"'" and candidate[-1] == candidate[0]:
        return candidate[1:-1]
    return part


def _safe_div(numerator, denominator):
    """Rate helper. Returns None rather than 0.0 when undefined, so that 'no data'
    is never silently reported as 'scored zero'."""
    if denominator == 0:
        return None
    return numerator / denominator


def _label_stratum(label_set: set) -> str:
    if len(label_set) == 0:
        return "blank"
    if len(label_set) == 1:
        return "single_key"
    return "multi_key"


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
def _metrics(records) -> dict:
    """Core metric block over a list of per-row outcome dicts."""
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    abstained = sum(1 for r in records if r["abstained"])
    wrong = sum(1 for r in records if r["wrong"])
    predicted = total - abstained

    return {
        "total": total,
        "predicted": predicted,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "match_rate": _safe_div(correct, total),
        "match_precision": _safe_div(correct, predicted),
        "abstention_rate": _safe_div(abstained, total),
    }


def score(predictions_df, solution_path) -> dict:
    """Score a predictions DataFrame against the BenchRec solution file.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Must contain columns `B_id` and `targetAllocation`. Extra columns ignored.
    solution_path : str
        Path to BenchRec_cash_v1.0_solution.csv (columns B_id, targetAllocation, Usage).

    Returns
    -------
    dict with overall metrics, a `by_label_type` breakdown (single_key / multi_key /
    blank), and data-integrity counts (missing B_ids, extras, duplicates).

    Scoring rules
    -------------
    * correct    : parsed prediction set == parsed label set, exactly.
    * abstention : parsed prediction set is empty AND the label set is non-empty.
    * blank label: an empty label set is scored, not dropped. It is correct only when
                   the prediction is also empty.
    * A solution B_id absent from the predictions is treated as an abstention.

    Note that `predicted` = total - abstained. A correct empty prediction on a blank
    label is therefore counted as a prediction, not an abstention — that follows from
    the rule that abstention requires a non-empty label.
    """
    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Solution file not found: {solution_path}")

    solution = pd.read_csv(solution_path, dtype=str, keep_default_na=False)

    for col in ("B_id", "targetAllocation"):
        if col not in solution.columns:
            raise ValueError(f"Solution file is missing required column '{col}'. "
                             f"Found: {list(solution.columns)}")
        if col not in predictions_df.columns:
            raise ValueError(f"Predictions are missing required column '{col}'. "
                             f"Found: {list(predictions_df.columns)}")

    preds = predictions_df[["B_id", "targetAllocation"]].copy()
    preds["B_id"] = preds["B_id"].astype(str).str.strip()

    n_pred_rows = len(preds)
    n_dupes = int(preds["B_id"].duplicated().sum())
    if n_dupes:
        preds = preds.drop_duplicates(subset="B_id", keep="first")

    pred_map = dict(zip(preds["B_id"], preds["targetAllocation"]))

    sol_ids = solution["B_id"].astype(str).str.strip()
    sol_id_set = set(sol_ids)
    missing_ids = sol_id_set - set(pred_map)
    extra_ids = set(pred_map) - sol_id_set

    records = []
    for b_id, raw_label in zip(sol_ids, solution["targetAllocation"]):
        label_set = _parse_alloc(raw_label)

        is_missing = b_id not in pred_map
        pred_set = set() if is_missing else _parse_alloc(pred_map[b_id])

        correct = pred_set == label_set
        abstained = (len(pred_set) == 0) and (len(label_set) > 0)
        wrong = (not correct) and (not abstained)

        records.append({
            "B_id": b_id,
            "stratum": _label_stratum(label_set),
            "correct": correct,
            "abstained": abstained,
            "wrong": wrong,
            "missing": is_missing,
        })

    result = _metrics(records)
    result["by_label_type"] = {
        stratum: _metrics([r for r in records if r["stratum"] == stratum])
        for stratum in ("single_key", "multi_key", "blank")
    }
    result["solution_path"] = solution_path
    result["prediction_rows_supplied"] = n_pred_rows
    result["duplicate_prediction_b_ids"] = n_dupes
    result["missing_b_ids"] = len(missing_ids)
    result["missing_b_ids_scored_as_abstentions"] = len(missing_ids)
    result["extra_prediction_b_ids_ignored"] = len(extra_ids)
    return result


# ----------------------------------------------------------------------------------
# Reporting helpers (private)
# ----------------------------------------------------------------------------------
def _pct(x):
    return "n/a" if x is None else f"{100.0 * x:.4f}%"


def _print_block(title, m, indent=""):
    print(f"{indent}{title}")
    print(f"{indent}  total rows        {m['total']:>10,}")
    print(f"{indent}  predicted         {m['predicted']:>10,}   (non-abstained)")
    print(f"{indent}  correct           {m['correct']:>10,}")
    print(f"{indent}  wrong             {m['wrong']:>10,}")
    print(f"{indent}  abstained         {m['abstained']:>10,}")
    print(f"{indent}  match rate        {_pct(m['match_rate']):>10}   (correct / total)")
    print(f"{indent}  match precision   {_pct(m['match_precision']):>10}   (correct / predicted)")
    print(f"{indent}  abstention rate   {_pct(m['abstention_rate']):>10}")


def _print_result(result):
    print("=" * 88)
    print("SCORE — MatcherByChatGPT_submission.csv vs BenchRec_cash_v1.0_solution.csv")
    print("=" * 88)
    print()
    _print_block("OVERALL", result)
    print()

    print("-" * 88)
    print("BROKEN OUT BY LABEL TYPE")
    print("-" * 88)
    labels = {
        "single_key": "SINGLE-KEY LABELS  (label set size == 1)",
        "multi_key": "MULTI-KEY LABELS   (label set size >= 2)",
        "blank": "BLANK LABELS       (label set empty — correct answer is 'no match')",
    }
    for key, title in labels.items():
        print()
        _print_block(title, result["by_label_type"][key])

    print()
    print("-" * 88)
    print("SIDE-BY-SIDE")
    print("-" * 88)
    rows = []
    for key in ("single_key", "multi_key", "blank"):
        m = result["by_label_type"][key]
        rows.append({
            "label_type": key,
            "total": m["total"],
            "predicted": m["predicted"],
            "correct": m["correct"],
            "wrong": m["wrong"],
            "abstained": m["abstained"],
            "match_rate": _pct(m["match_rate"]),
            "match_precision": _pct(m["match_precision"]),
            "abstention_rate": _pct(m["abstention_rate"]),
        })
    rows.append({
        "label_type": "ALL",
        "total": result["total"],
        "predicted": result["predicted"],
        "correct": result["correct"],
        "wrong": result["wrong"],
        "abstained": result["abstained"],
        "match_rate": _pct(result["match_rate"]),
        "match_precision": _pct(result["match_precision"]),
        "abstention_rate": _pct(result["abstention_rate"]),
    })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("-" * 88)
    print("DATA INTEGRITY")
    print("-" * 88)
    print(f"  prediction rows supplied            {result['prediction_rows_supplied']:>10,}")
    print(f"  duplicate B_ids in predictions      {result['duplicate_prediction_b_ids']:>10,}"
          "   (first kept)")
    print(f"  solution B_ids MISSING from preds   {result['missing_b_ids']:>10,}"
          "   (scored as abstentions)")
    print(f"  extra prediction B_ids not in sol   {result['extra_prediction_b_ids_ignored']:>10,}"
          "   (ignored)")
    if result["missing_b_ids"] == 0:
        print()
        print("  No solution B_ids are missing from the prediction file — nothing was")
        print("  imputed as an abstention on account of absence.")


def _temporal_check(data_dir):
    print()
    print("=" * 88)
    print("B_valueDate RANGE — TRAIN vs EVAL")
    print("=" * 88)

    files = {
        "train": os.path.join(data_dir, "BenchRec_cash_v1.0_train.csv"),
        "eval": os.path.join(data_dir, "BenchRec_cash_v1.0_eval.csv"),
    }

    spans = {}
    rows = []
    for tag, path in files.items():
        if not os.path.exists(path):
            print(f"  {tag}: file not found at {path}")
            continue
        df = pd.read_csv(path, usecols=["B_valueDate"], dtype=str, keep_default_na=False)
        dates = pd.to_datetime(df["B_valueDate"].replace("", None),
                               errors="coerce", format="mixed").dropna()
        spans[tag] = dates
        rows.append({
            "file": tag,
            "non_null_B_valueDate": len(dates),
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
            "span_days": int((dates.max() - dates.min()).days),
            "p01": str(dates.quantile(0.01).date()),
            "median": str(dates.quantile(0.50).date()),
            "p99": str(dates.quantile(0.99).date()),
        })

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  min/max are the literal extremes you asked for; p01/median/p99 are included")
    print("  beside them because the extremes are not representative — see below.")

    if spans:
        print()
        print("  Row counts by year:")
        year_tab = pd.DataFrame({
            tag: dates.dt.year.value_counts().sort_index()
            for tag, dates in spans.items()
        }).fillna(0).astype(int)
        year_tab.index.name = "year"
        print(year_tab.to_string())

    if "train" not in spans or "eval" not in spans:
        return

    tr, ev = spans["train"], spans["eval"]
    tr_min, tr_max = tr.min(), tr.max()
    ev_min, ev_max = ev.min(), ev.max()

    print()
    print("-" * 88)
    print("IS THE SPLIT TEMPORAL?")
    print("-" * 88)
    print()

    overlap_lo, overlap_hi = max(tr_min, ev_min), min(tr_max, ev_max)
    disjoint = ev_min > tr_max

    ev_before_train_max = float((ev <= tr_max).mean())
    tr_after_eval_min = float((tr >= ev_min).mean())

    print(f"  train B_valueDate   {tr_min.date()}  ..  {tr_max.date()}")
    print(f"  eval  B_valueDate   {ev_min.date()}  ..  {ev_max.date()}")
    print()
    print(f"  eval rows dated on or before train's last date   "
          f"{ev_before_train_max * 100:.4f}%")
    print(f"  train rows dated on or after eval's first date   "
          f"{tr_after_eval_min * 100:.4f}%")
    print()

    n_ev_before = int((ev <= tr_max).sum())

    if disjoint:
        gap = (ev_min - tr_max).days
        print("  VERDICT: YES — this is a clean temporal split.")
        print(f"  Every eval transaction is dated after every train transaction, with a")
        print(f"  {gap}-day gap between train's last date ({tr_max.date()}) and eval's")
        print(f"  first ({ev_min.date()}).")
    elif ev_min < tr_min:
        print("  VERDICT: NO — eval PRECEDES train, so the split runs backwards in time.")
    elif ev_before_train_max <= 0.05:
        print("  VERDICT: YES, effectively — this is a temporal split with a thin overlap tail.")
        print()
        print(f"  The ranges technically overlap ({overlap_lo.date()} .. {overlap_hi.date()}, "
              f"{(overlap_hi - overlap_lo).days} days),")
        print(f"  but only {n_ev_before:,} eval rows ({ev_before_train_max * 100:.4f}%) fall at or")
        print(f"  before train's last date. The other "
              f"{(1 - ev_before_train_max) * 100:.4f}% come strictly after it.")
        print(f"  Read the split as: train up to ~{tr_max.date()}, evaluate on what follows,")
        print(f"  plus a handful of late-settling stragglers.")
        print()
        print("  Do not be misled by train's min. Its range LOOKS like 8 years, but the mass")
        print("  is not spread over it — see the by-year counts above. Train is essentially")
        print("  2022 through early 2023, with a few outlier rows dated much earlier.")
    else:
        overlap_days = (overlap_hi - overlap_lo).days
        print("  VERDICT: NO — not a temporal split. The two ranges OVERLAP substantially.")
        print(f"  Overlap window: {overlap_lo.date()} .. {overlap_hi.date()} "
              f"({overlap_days} days).")
        print(f"  {ev_before_train_max * 100:.4f}% of eval rows fall inside the period train")
        print("  already covers, so the split is at least partly random rather than by time.")

    print()
    print("  Consequence either way: allocation keys embed a date (field 2 of the key), so a")
    print("  key observed in train can essentially never be reused verbatim in eval. Memorising")
    print("  the training label set will not transfer, and a random-shuffle validation split of")
    print("  train will read optimistically against eval.")


def _main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    solution_path = os.path.join(data_dir, "BenchRec_cash_v1.0_solution.csv")
    submission_path = os.path.join(data_dir, "MatcherByChatGPT_submission.csv")

    predictions_df = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    result = score(predictions_df, solution_path)
    _print_result(result)
    _temporal_check(data_dir)


if __name__ == "__main__":
    _main()
