"""
Build one dataset from our own runs and last year's runs.

Our runs: the harness (harness/components/extract_run.py) writes two small CSV files
per run into harness/results/<run_id>-results/:
    <run_id>_runs.csv         one row: hyperparameters, steps, time, rewards
    <run_id>_progression.csv  reward curve (one row per summary step)

This script:
  1. collects all those files into runs.csv and progression.csv (next to the output file)
  2. computes reward_at_500k for every run from its reward curve, the same way as
     for last year's data (mean of the last 3 summaries with step <= 515k;
     the run is usable only if the last of them is at >= 485k steps)
  3. adds data/previous_year_runs.csv and writes everything to data/all_runs.csv,
     with the same columns and the same value formats

After this, normalise all runs together:
    python data_collection/normalize_rewards.py data/all_runs.csv --column reward_at_500k --only-usable --clip

Usage (from the ml-agents root):
    python data_collection/build_dataset.py
    python data_collection/build_dataset.py --results harness/results --previous data/previous_year_runs.csv
"""

import argparse
import csv
import glob
import math
import os
import sys

from standardize_previous import COLUMNS, reward_at_budget, to_float

OUR_GROUP = "7 (2026)"

# extra columns that only our runs have
ALL_COLUMNS = COLUMNS + ["seed", "mean_reward_last3", "steps_per_update", "buffer_init_steps"]

PROGRESSION_COLUMNS = ["run_id", "step", "cumulative_reward", "episode_length",
                       "entropy", "learning_rate"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_original(path):
    """True if the file is in the folder the harness made (<run_id>-results), not a copy."""
    folder = os.path.basename(os.path.dirname(path))
    run_id = os.path.basename(path)[:-len("_runs.csv")]
    return folder == run_id + "-results"


def collect(results_dir):
    """All per-run files from the harness -> (run rows, progression rows)."""
    runs, progression = [], []
    run_columns, prog_columns = [], None
    seen = set()
    paths = glob.glob(os.path.join(glob.escape(results_dir), "*", "*_runs.csv"))
    paths.sort(key=lambda p: (not is_original(p), p))   # original folders first, copies after
    for path in paths:
        rows = read_csv(path)
        if not rows or "run_id" not in rows[0]:
            print(f"skip   {path} (empty or not a harness file)")
            continue
        if rows[0]["run_id"] in seen:
            print(f"skip   {path} (copy of run {rows[0]['run_id']}, already read)")
            continue
        seen.add(rows[0]["run_id"])
        runs += rows
        for col in rows[0].keys():          # union of columns, older files may have fewer
            if col not in run_columns:
                run_columns.append(col)

        prog_path = path[:-len("_runs.csv")] + "_progression.csv"
        if os.path.exists(prog_path):
            prog = read_csv(prog_path)
            progression += prog
            if prog:
                prog_columns = prog_columns or list(prog[0].keys())
        else:
            print(f"warning: no progression file for {rows[0]['run_id']}")
    return runs, progression, run_columns, prog_columns


def same_format(row):
    """Write values the same way for every group (True/False, plain numbers)."""
    for key, value in row.items():
        if value is None:
            row[key] = ""
            continue
        value = str(value).strip()
        if value.lower() in ("true", "false"):
            value = value.capitalize()          # TRUE / true -> True
        elif key not in ("source_group", "source_run_id") and to_float(value) is not None:
            number = to_float(value)
            if math.isfinite(number) and number == int(number) and "e" not in value.lower() and abs(number) < 1e15:
                value = str(int(number))        # 492000.0 -> 492000
        row[key] = value
    return row


def our_rows(runs, progression):
    """Our runs in the same columns as last year's data."""
    curves = {}
    for p in progression:
        step, reward = to_float(p.get("step")), to_float(p.get("cumulative_reward"))
        if step is None or reward is None or not math.isfinite(step) or not math.isfinite(reward):
            continue                        # empty / nan / inf point -> not used
        curves.setdefault(p["run_id"], []).append((step, reward, None))

    rows = []
    for r in runs:
        reward_500k, _ = reward_at_budget(curves.get(r["run_id"], []))
        row = {
            "source_group": OUR_GROUP,
            "source_run_id": r["run_id"],
            "final_reward": r.get("cumulative_reward_mean", ""),
            "reward_at_500k": "" if reward_500k is None else reward_500k,
            "usable_500k": reward_500k is not None,
        }
        for col in ALL_COLUMNS:
            if col not in row and col in r:
                row[col] = r[col]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Combine our runs and last year's runs into one CSV.")
    parser.add_argument("--results", default=os.path.join("harness", "results"))
    parser.add_argument("--previous", default=os.path.join("data", "previous_year_runs.csv"))
    parser.add_argument("--out", default=os.path.join("data", "all_runs.csv"))
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        sys.exit(f"error: results folder {args.results} not found (run this from the ml-agents root)")
    if not os.path.isfile(args.previous):
        sys.exit(f"error: previous-year file {args.previous} not found or not a file")

    # 1) our runs
    runs, progression, run_columns, prog_columns = collect(args.results)
    out_dir = os.path.dirname(args.out) or "."
    write_csv(os.path.join(out_dir, "runs.csv"), run_columns or ["run_id"], runs)
    write_csv(os.path.join(out_dir, "progression.csv"), prog_columns or PROGRESSION_COLUMNS, progression)
    print(f"our runs: {len(runs)} (progression rows: {len(progression)})")
    if runs and not progression:
        print("warning: no reward curves found (is tensorboard installed?) -> none of our runs is usable at 500k")

    # 2) reward at 500k for our runs
    ours = our_rows(runs, progression)
    usable = sum(1 for r in ours if r["usable_500k"])
    print(f"  usable at 500k: {usable}")

    # 3) last year's runs
    previous = read_csv(args.previous)
    print(f"previous-year runs: {len(previous)} "
          f"(usable: {sum(1 for r in previous if r['usable_500k'].lower() == 'true')})")

    rows = [same_format(r) for r in ours + previous]
    write_csv(args.out, ALL_COLUMNS, rows)
    print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
