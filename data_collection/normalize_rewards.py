"""
Min-max normalise the reward per environment and algorithm.

Worm and Walker give rewards on very different scales, so before comparing them
(e.g. the transfer test Worm -> Walker) we put every reward between 0 and 1:

    reward_norm = (reward - min) / (max - min)

min and max are taken from all runs with the same environment AND algorithm
(Worm-PPO, Worm-SAC, Walker-PPO, Walker-SAC), like described in our Phase 2 plan.

Usage:
    python data_collection/normalize_rewards.py data/runs.csv --column mean_reward_last3
    python data_collection/normalize_rewards.py data/previous_year_runs.csv --column reward_at_500k --only-usable

With --clip, min and max are replaced by the outlier limits of the IQR rule
(Q1 - 1.5*IQR and Q3 + 1.5*IQR). A run above the upper limit gets 1.0 instead of
squeezing all other runs towards 0. The run itself is NOT removed.

Writes a new file next to the input (<name>_normalized.csv) with an extra column
<column>_norm, and prints the min/max that were used for every group.
"""

import argparse
import csv
import os
import statistics


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Min-max normalise rewards per environment and algorithm.")
    parser.add_argument("input", help="CSV file, e.g. data/runs.csv")
    parser.add_argument("--column", default="mean_reward_last3", help="reward column to normalise")
    parser.add_argument("--only-usable", action="store_true",
                        help="only normalise rows with usable_500k = True, other rows stay with an empty value")
    parser.add_argument("--clip", action="store_true",
                        help="use IQR outlier limits instead of plain min/max")
    parser.add_argument("--out", help="output file (default: <input>_normalized.csv)")
    args = parser.parse_args()

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    # keep all rows in the output, but only use the usable ones for min/max
    all_rows = rows
    if args.only_usable:
        rows_to_normalize = [r for r in all_rows if r.get("usable_500k") == "True"]
    else:
        rows_to_normalize = all_rows

    # 1) collect all rewards per (environment, algorithm)
    groups = {}
    for r in rows_to_normalize:
        value = to_float(r[args.column])
        if value is None:
            continue
        key = (r["environment"], r["drl_algorithm"].upper())
        groups.setdefault(key, []).append(value)

    # 2) min and max per group (or the IQR limits with --clip)
    limits = {}
    print(f"{'environment':12} {'algorithm':10} {'runs':>5} {'low':>10} {'high':>10}")
    for key in sorted(groups):
        values = groups[key]
        low, high = min(values), max(values)
        if args.clip and len(values) >= 4:      # IQR needs a few values to make sense
            q1, _, q3 = statistics.quantiles(values, n=4)
            iqr = q3 - q1
            low = max(low, q1 - 1.5 * iqr)      # never go outside the real min/max
            high = min(high, q3 + 1.5 * iqr)
        limits[key] = (low, high)
        print(f"{key[0]:12} {key[1]:10} {len(values):5} {low:10.3f} {high:10.3f}")

    # 3) normalise every row with the min/max of its own group
    new_column = args.column + "_norm"
    skipped = 0
    clipped = 0
    for r in all_rows:
        if args.only_usable and r.get("usable_500k") != "True":
            r[new_column] = ""    # not usable -> no normalised value
            continue
        value = to_float(r[args.column])
        key = (r["environment"], r["drl_algorithm"].upper())
        if value is None or key not in limits:
            r[new_column] = ""
            continue
        low, high = limits[key]
        if high == low:
            r[new_column] = ""    # only one run (or all equal) -> cannot normalise
            skipped += 1
            continue
        norm = (value - low) / (high - low)
        if norm > 1 or norm < 0:
            clipped += 1
            norm = min(max(norm, 0), 1)         # outlier -> 1.0 (or 0.0)
        r[new_column] = round(norm, 4)

    if clipped:
        print(f"\n{clipped} outlier(s) clipped to 0 or 1 (still in the data)")
    if skipped:
        print(f"\nwarning: {skipped} row(s) not normalised, their group has only one value")

    out = args.out or os.path.splitext(args.input)[0] + "_normalized.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns + [new_column])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n{len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
