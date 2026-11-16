"""
Extracts data for a SINGLE training run and outputs unique CSV files to avoid race conditions.
Returns exit code 0 on success (so the runner can clean up) or 1 on failure.
"""

import csv
import json
import os
import sys
import yaml
from datetime import datetime, timezone

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None

RUN_COLUMNS = [
    "run_id", "timestamp", "environment", "drl_algorithm", "seed",
    "torch_version", "mlagents_version",
    "batch_size", "buffer_size", "learning_rate", "learning_rate_schedule",
    "hidden_units", "num_layers", "normalize", "gamma", "max_steps", "time_horizon",
    "beta", "epsilon", "lambda", "num_epoch",
    "tau", "steps_per_update", "init_entcoef", "buffer_init_steps",
    "total_steps", "time_elapsed_seconds",
    "cumulative_reward_mean", "max_reward", "mean_reward_last3"
]

PROGRESSION_COLUMNS = ["run_id", "step", "cumulative_reward", "episode_length",
                       "entropy", "learning_rate"]

TB_TAGS = {
    "Environment/Cumulative Reward": "cumulative_reward",
    "Environment/Episode Length": "episode_length",
    "Policy/Entropy": "entropy",
    "Policy/Learning Rate": "learning_rate",
}

def read_config(run_dir):
    with open(os.path.join(run_dir, "configuration.yaml")) as f:
        config = yaml.safe_load(f)

    environment, behavior = next(iter(config["behaviors"].items()))
    hp = behavior["hyperparameters"]
    net = behavior["network_settings"]

    return environment, {
        "drl_algorithm": behavior["trainer_type"].upper(),
        "seed": config["env_settings"]["seed"],
        "batch_size": hp.get("batch_size"),
        "buffer_size": hp.get("buffer_size"),
        "learning_rate": hp.get("learning_rate"),
        "learning_rate_schedule": hp.get("learning_rate_schedule"),
        "hidden_units": net.get("hidden_units"),
        "num_layers": net.get("num_layers"),
        "normalize": net.get("normalize"),
        "gamma": behavior["reward_signals"]["extrinsic"]["gamma"],
        "max_steps": behavior.get("max_steps"),
        "time_horizon": behavior.get("time_horizon"),
        "beta": hp.get("beta"),
        "epsilon": hp.get("epsilon"),
        "lambda": hp.get("lambd"),
        "num_epoch": hp.get("num_epoch"),
        "tau": hp.get("tau"),
        "steps_per_update": hp.get("steps_per_update"),
        "init_entcoef": hp.get("init_entcoef"),
        "buffer_init_steps": hp.get("buffer_init_steps"),
    }

def read_timers(run_dir, environment):
    with open(os.path.join(run_dir, "run_logs", "timers.json")) as f:
        timers = json.load(f)

    meta = timers["metadata"]
    start = datetime.fromtimestamp(int(meta["start_time_seconds"]), tz=timezone.utc)
    
    reward = timers["gauges"].get(f"{environment}.Environment.CumulativeReward.mean", {})
    if not reward:
        for k, v in timers["gauges"].items():
            if "CumulativeReward" in k:
                reward = v
                break

    return {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_elapsed_seconds": round(timers["total"], 2),
        "torch_version": meta.get("pytorch_version"),
        "mlagents_version": meta.get("mlagents_version"),
        "cumulative_reward_mean": reward.get("value"),
        "max_reward": reward.get("max"),
    }

def read_total_steps(run_dir, environment):
    with open(os.path.join(run_dir, "run_logs", "training_status.json")) as f:
        status = json.load(f)
    try:
        return status[environment]["final_checkpoint"]["steps"]
    except KeyError:
        for k, v in status.items():
            if isinstance(v, dict) and "final_checkpoint" in v:
                return v["final_checkpoint"]["steps"]
        return 0

def read_progression(run_dir, environment):
    if EventAccumulator is None:
        return []
    events = EventAccumulator(os.path.join(run_dir, environment))
    events.Reload()

    rows = {}
    available = events.Tags()["scalars"]
    for tag, column in TB_TAGS.items():
        if tag not in available:
            continue
        for e in events.Scalars(tag):
            rows.setdefault(e.step, {"step": e.step})[column] = e.value
    return [rows[s] for s in sorted(rows)]

def extract_run(run_dir):
    run_id = os.path.basename(os.path.normpath(run_dir))
    environment, hyperparameters = read_config(run_dir)

    row = {"run_id": run_id, "environment": environment}
    row.update(hyperparameters)
    row.update(read_timers(run_dir, environment))
    row["total_steps"] = read_total_steps(run_dir, environment)

    progression = read_progression(run_dir, environment)
    rewards = [p["cumulative_reward"] for p in progression if "cumulative_reward" in p]
    if rewards:
        row["cumulative_reward_mean"] = rewards[-1]
        row["max_reward"] = max(rewards)
        last3 = rewards[-3:]
        row["mean_reward_last3"] = sum(last3) / len(last3)

    for p in progression:
        p["run_id"] = run_id
    return row, progression

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m harness.components.extract_run <run_dir>")
        sys.exit(1)
        
    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"Error: Directory {run_dir} not found.")
        sys.exit(1)

    run_id = os.path.basename(os.path.normpath(run_dir))
    results_dir = os.path.join("harness", "results", f"{run_id}-results")
    os.makedirs(results_dir, exist_ok=True)
    
    runs_csv_path = os.path.join(results_dir, f"{run_id}_runs.csv")
    progression_csv_path = os.path.join(results_dir, f"{run_id}_progression.csv")

    try:
        row, progression = extract_run(run_dir)
    except Exception as e:
        print(f"Error extracting {run_id}: {e!r}")
        sys.exit(1)

    with open(runs_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    if progression:
        with open(progression_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROGRESSION_COLUMNS)
            writer.writeheader()
            writer.writerows(progression)

    print(f"Extraction successful: {run_id}")
    sys.exit(0)

if __name__ == "__main__":
    main()