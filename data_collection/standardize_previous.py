"""
Bring last year's Worm/Walker data from the shared data-collection repo
into one table with our column names.

Groups used (AY 2025-2026): 1, 14, 15, 16. Every group stored its data differently,
so there is one small function per group.

Our runs use a budget of 500,000 steps. Most old runs trained much longer, so we
also compute the reward at 500k steps from the reward curve, if the group saved one.
Runs where that is possible get usable_500k = True.

Usage:
    python data_collection/standardize_previous.py "path/to/BCS2720-project-data-collection/AY 2025-2026"
"""

import csv
import os
import sys
import zipfile
import io

BUDGET = 500000
TOLERANCE = 1.03   # accept summary steps up to 515,000 (groups log every 10k-30k steps)

COLUMNS = [
    "source_group", "source_run_id", "environment", "drl_algorithm",
    "cpu_model", "cpu_cores", "cpu_threads", "total_ram_gb", "gpu_model", "operating_system",
    "batch_size", "buffer_size", "learning_rate", "learning_rate_schedule",
    "hidden_units", "num_layers", "normalize", "gamma", "max_steps", "time_horizon",
    "beta", "epsilon", "lambda", "num_epoch", "tau", "init_entcoef",
    "total_steps", "time_elapsed_seconds", "final_reward", "max_reward",
    "reward_at_500k", "time_at_500k_seconds", "peak_ram_mb", "avg_ram_mb", "usable_500k",
]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def reward_at_budget(curve):
    """curve = list of (step, reward, time). Returns mean of the last 3 rewards
    with step <= 500k (same idea as our mean_reward_last3) and the time at that point."""
    points = [p for p in curve if p[0] is not None and p[1] is not None and p[0] <= BUDGET * TOLERANCE]
    points.sort()
    if len(points) < 3 or points[-1][0] < BUDGET * 0.97:
        return None, None      # run did not get (close) to 500k steps
    last3 = points[-3:]
    mean = sum(p[1] for p in last3) / 3
    return round(mean, 4), last3[-1][2]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Group 1
# one row per run, no reward curve -> reward at 500k not possible
def group1(root):
    rows = []
    for r in read_csv(os.path.join(root, "Group 1", "Results", "Complete_dataset.csv")):
        if r["env_name"] not in ("Worm", "Walker"):
            continue
        rows.append({
            "source_group": "1", "source_run_id": r["run_id"],
            "environment": r["env_name"], "drl_algorithm": r["algo_name"].upper(),
            "cpu_threads": r["cpu_cores_logical"], "total_ram_gb": r["ram_total_gb"],
            "batch_size": r["batch_size"], "learning_rate": r["learning_rate"],
            "num_layers": r["nn_arch_depth"],
            "time_elapsed_seconds": r["wallclock_seconds_total"],
            "final_reward": r["final_perf"],
            "usable_500k": False,
        })
    return rows


# ---------------------------------------------------------------- Group 14
# one file per run with the reward every 30k steps + some hyperparameters
def group14(root):
    rows = []
    folder = os.path.join(root, "Group 14")
    for name in sorted(os.listdir(folder)):
        if not name.lower().startswith("training_data_worm") or not name.endswith(".csv"):
            continue
        data = read_csv(os.path.join(folder, name))
        first = data[0]
        if "batch_size" not in first:
            continue           # the two oldest files have no hyperparameters, so no features
        curve = [(to_float(d["step"]), to_float(d["mean_reward"]), to_float(d["time_elapsed"])) for d in data]
        rewards = [c[1] for c in curve if c[1] is not None]
        r500, t500 = reward_at_budget(curve)
        last = data[-1]
        rows.append({
            "source_group": "14", "source_run_id": name[len("training_data_"):-4],
            "environment": "Worm", "drl_algorithm": first["trainer_type"].upper(),
            "batch_size": first["batch_size"], "buffer_size": first["buffer_size"],
            "learning_rate": first["learning_rate"], "learning_rate_schedule": first["learning_rate_schedule"],
            "beta": first["beta"], "epsilon": first["epsilon"], "lambda": first["lambd"],
            "num_epoch": first["num_epoch"],
            "total_steps": last["step"], "time_elapsed_seconds": last["time_elapsed"],
            "final_reward": last["mean_reward"], "max_reward": max(rewards),
            "reward_at_500k": r500, "time_at_500k_seconds": t500,
            "peak_ram_mb": max(to_float(d["memory_usage_peak_mb"]) or 0 for d in data),
            "avg_ram_mb": last["memory_usage_avg_mb"],
            "usable_500k": r500 is not None,
        })
    return rows


# ---------------------------------------------------------------- Group 15
# zip with static.csv (settings + hardware), summary.csv (results), training.csv (curves)
def group15(root):
    with zipfile.ZipFile(os.path.join(root, "Group 15", "Group15_data.zip")) as z:
        def zcsv(name):
            path = [n for n in z.namelist() if n.endswith(name)][0]
            return list(csv.DictReader(io.TextIOWrapper(z.open(path), encoding="utf-8-sig")))
        static = [s for s in zcsv("static.csv") if s["game_name"] in ("Worm", "Walker")]
        ids = {s["RunID"] for s in static}
        summary = {s["RunID"]: s for s in zcsv("summary.csv") if s["RunID"] in ids}
        curves = {}
        for t in zcsv("training.csv"):
            if t["RunID"] in ids:
                curves.setdefault(t["RunID"], []).append(
                    (to_float(t["Step"]), to_float(t["cumulative_reward"]), to_float(t["time_elapsed(s)"])))

    rows = []
    for s in static:
        sm = summary.get(s["RunID"], {})
        curve = curves.get(s["RunID"], [])
        r500, t500 = reward_at_budget(curve)
        steps = max((c[0] for c in curve if c[0] is not None), default=None)
        ram = to_float(s["system_ram_gb"])
        if ram and ram > 10000:   # this group saved RAM in bytes, not GB
            ram = round(ram / 1024 ** 3, 2)
        rows.append({
            "source_group": "15", "source_run_id": s["RunID"],
            "environment": s["game_name"], "drl_algorithm": s["training_type"].upper(),
            "cpu_model": s["cpu"], "cpu_cores": s["physical_cores"], "total_ram_gb": ram,
            "gpu_model": s["gpu"], "operating_system": s["os"],
            "batch_size": s["batch_size"], "buffer_size": s["buffer_size"],
            "learning_rate": s["learning_rate"], "learning_rate_schedule": s["learning_rate_schedule"],
            "hidden_units": s["hidden_units"], "num_layers": s["num_layers"], "normalize": s["normalize"],
            "gamma": s["reward_extrinsic_gamma"], "max_steps": s["max_steps"], "time_horizon": s["time_horizon"],
            "beta": s["beta"], "epsilon": s["epsilon"], "lambda": s["lambd"], "num_epoch": s["num_epoch"],
            "total_steps": steps, "time_elapsed_seconds": sm.get("total_duration_sec"),
            "final_reward": sm.get("final_reward"), "max_reward": sm.get("max_reward"),
            "reward_at_500k": r500, "time_at_500k_seconds": t500,
            "peak_ram_mb": sm.get("max_ram_used_mb"), "avg_ram_mb": sm.get("mean_ram_used_mb"),
            "usable_500k": r500 is not None,
        })
    return rows


# ---------------------------------------------------------------- Group 16
# one row per run (trained ~3M steps), no reward curve -> reward at 500k not possible
def group16(root):
    rows = []
    for r in read_csv(os.path.join(root, "Group 16", "data.csv")):
        if r["enviroment"] not in ("Worm", "Walker"):     # their column name has a typo
            continue
        rows.append({
            "source_group": "16", "source_run_id": r["timestamp"],
            "environment": r["enviroment"], "drl_algorithm": r["algorithm"].upper(),
            "cpu_cores": r["cpu_cores"], "total_ram_gb": r["ram_gb"],
            "gpu_model": r["gpu_name"], "operating_system": r["os"],
            "batch_size": r["batch_size"], "buffer_size": r["buffer_size"],
            "learning_rate": r["learning_rate"], "hidden_units": r["hidden_units"],
            "num_layers": r["num_layers"], "gamma": r["gamma"], "max_steps": r["max_steps"],
            "time_horizon": r["time_horizon"], "beta": r["beta"], "epsilon": r["epsilon"],
            "lambda": r["lambd"], "num_epoch": r["num_epoch"],
            "tau": r["tau"] if r["algorithm"] == "sac" else "",
            "init_entcoef": r["init_entcoef"] if r["algorithm"] == "sac" else "",
            "total_steps": r["steps"], "time_elapsed_seconds": r["training_duration_seconds"],
            "final_reward": r["final_cumulative_mean_reward"], "peak_ram_mb": r["peak_ram_mb"],
            "usable_500k": False,
        })
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "AY 2025-2026"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/previous_year_runs.csv"

    rows = []
    for name, func in (("Group 1", group1), ("Group 14", group14), ("Group 15", group15), ("Group 16", group16)):
        try:
            part = func(root)
        except (OSError, KeyError, IndexError) as e:
            print(f"error  {name}: {e!r}")
            continue
        usable = sum(1 for r in part if r["usable_500k"])
        print(f"{name}: {len(part)} Worm/Walker runs, {usable} usable at 500k steps")
        rows += part

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} runs -> {out} ({sum(1 for r in rows if r['usable_500k'])} usable at 500k)")


if __name__ == "__main__":
    main()
