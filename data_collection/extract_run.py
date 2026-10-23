"""
Extract one CSV row per ML-Agents training run.

Reads a results folder created by `mlagents-learn` (results/<run_id>/) and writes:
  - data/runs.csv         one row per run: hardware, hyperparameters, training time, performance
  - data/progression.csv  one row per summary step: reward curve of every run

Usage (from the ml-agents root, with the venv active):
    python data_collection/extract_run.py results/worm_test1 results/worm_lr1e-3
    python data_collection/extract_run.py --all results

Runs that are already in runs.csv (same run_id) are skipped, so the script can be
re-run safely after every new training run.

Note: hardware columns describe the machine the script runs on, so run it on the
same machine that did the training.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import yaml

try:
    import psutil
except ImportError:
    psutil = None

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


# Column order of runs.csv. Column names follow the naming used in the shared
# BCS2720 data-collection repo, so our data can be combined with the data that
# other groups share this year.
RUN_COLUMNS = [
    # identification
    "run_id", "timestamp", "environment", "drl_algorithm", "seed",
    # hardware
    "cpu_model", "cpu_cores", "cpu_threads", "cpu_frequency_mhz", "total_ram_gb",
    "gpu_model", "operating_system", "torch_version", "mlagents_version",
    # hyperparameters (shared by PPO and SAC)
    "batch_size", "buffer_size", "learning_rate", "learning_rate_schedule",
    "hidden_units", "num_layers", "normalize", "gamma", "max_steps", "time_horizon",
    # PPO-only hyperparameters (empty for SAC)
    "beta", "epsilon", "lambda", "num_epoch",
    # SAC-only hyperparameters (empty for PPO)
    "tau", "steps_per_update", "init_entcoef", "buffer_init_steps",
    # results (targets)
    "total_steps", "time_elapsed_seconds",
    "cumulative_reward_mean", "max_reward", "mean_reward_last3",
]

PROGRESSION_COLUMNS = ["run_id", "step", "cumulative_reward", "episode_length",
                       "entropy", "learning_rate"]

# TensorBoard tag written by ML-Agents -> column name in progression.csv
TB_TAGS = {
    "Environment/Cumulative Reward": "cumulative_reward",
    "Environment/Episode Length": "episode_length",
    "Policy/Entropy": "entropy",
    "Policy/Learning Rate": "learning_rate",
}


# ---------------------------------------------------------------- hardware

def get_cpu_model():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_gpu_model():
    # Only NVIDIA GPUs are detected (via nvidia-smi); everything else is "none/unknown".
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:
        return "none/unknown"


def get_os_name():
    # platform.release() says "10" on Windows 11 too; build 22000+ means Windows 11.
    if platform.system() == "Windows":
        build = int(platform.version().split(".")[-1])
        return f"Windows {11 if build >= 22000 else 10} (build {build})"
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.release()}"


def get_hardware():
    hw = {
        "cpu_model": get_cpu_model(),
        "cpu_cores": None,
        "cpu_threads": os.cpu_count(),
        "cpu_frequency_mhz": None,
        "total_ram_gb": None,
        "gpu_model": get_gpu_model(),
        "operating_system": get_os_name(),
    }
    if psutil is not None:
        hw["cpu_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq is not None:
            hw["cpu_frequency_mhz"] = round(freq.max or freq.current)
        hw["total_ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 2)
    return hw


# ---------------------------------------------------------------- run folder

def read_config(run_dir):
    """Hyperparameters from configuration.yaml (ML-Agents writes ALL values here, incl. defaults)."""
    with open(os.path.join(run_dir, "configuration.yaml")) as f:
        config = yaml.safe_load(f)

    # one results folder = one behavior (e.g. "Worm")
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
        # PPO only
        "beta": hp.get("beta"),
        "epsilon": hp.get("epsilon"),
        "lambda": hp.get("lambd"),          # ML-Agents spells it "lambd"
        "num_epoch": hp.get("num_epoch"),
        # SAC only
        "tau": hp.get("tau"),
        "steps_per_update": hp.get("steps_per_update"),
        "init_entcoef": hp.get("init_entcoef"),
        "buffer_init_steps": hp.get("buffer_init_steps"),
    }


def read_timers(run_dir, environment):
    """Start time, total duration, versions and final reward gauges from run_logs/timers.json."""
    with open(os.path.join(run_dir, "run_logs", "timers.json")) as f:
        timers = json.load(f)

    meta = timers["metadata"]
    start = datetime.fromtimestamp(int(meta["start_time_seconds"]), tz=timezone.utc)
    reward = timers["gauges"].get(f"{environment}.Environment.CumulativeReward.mean", {})

    return {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_elapsed_seconds": round(timers["total"], 2),
        "torch_version": meta.get("pytorch_version"),
        "mlagents_version": meta.get("mlagents_version"),
        # fallback values, replaced by the TensorBoard curve if it can be read
        "cumulative_reward_mean": reward.get("value"),
        "max_reward": reward.get("max"),
    }


def read_total_steps(run_dir, environment):
    with open(os.path.join(run_dir, "run_logs", "training_status.json")) as f:
        status = json.load(f)
    return status[environment]["final_checkpoint"]["steps"]


def read_progression(run_dir, environment):
    """Reward curve: one dict per summary step, read from the TensorBoard event file."""
    if EventAccumulator is None:
        return []
    events = EventAccumulator(os.path.join(run_dir, environment))
    events.Reload()

    rows = {}  # step -> row
    available = events.Tags()["scalars"]
    for tag, column in TB_TAGS.items():
        if tag not in available:
            continue
        for e in events.Scalars(tag):
            rows.setdefault(e.step, {"step": e.step})[column] = e.value
    return [rows[s] for s in sorted(rows)]


def extract_run(run_dir, hardware):
    run_id = os.path.basename(os.path.normpath(run_dir))
    environment, hyperparameters = read_config(run_dir)

    row = {"run_id": run_id, "environment": environment}
    row.update(hardware)
    row.update(hyperparameters)
    row.update(read_timers(run_dir, environment))
    row["total_steps"] = read_total_steps(run_dir, environment)

    progression = read_progression(run_dir, environment)
    rewards = [p["cumulative_reward"] for p in progression if "cumulative_reward" in p]
    if rewards:
        row["cumulative_reward_mean"] = rewards[-1]           # reward at the last summary
        row["max_reward"] = max(rewards)
        last3 = rewards[-3:]
        row["mean_reward_last3"] = sum(last3) / len(last3)    # less noisy than one point

    for p in progression:
        p["run_id"] = run_id
    return row, progression


# ---------------------------------------------------------------- CSV output

def existing_run_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def append_rows(path, columns, rows, skip_run_ids=()):
    rows = [r for r in rows if r["run_id"] not in skip_run_ids]
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Turn ML-Agents results folders into CSV rows.")
    parser.add_argument("run_dirs", nargs="*", help="results/<run_id> folders")
    parser.add_argument("--all", metavar="RESULTS_DIR",
                        help="extract every run folder inside RESULTS_DIR")
    parser.add_argument("--out", default="data/runs.csv")
    parser.add_argument("--progression", default="data/progression.csv")
    args = parser.parse_args()

    run_dirs = list(args.run_dirs)
    if args.all:
        run_dirs += [os.path.join(args.all, d) for d in sorted(os.listdir(args.all))]
    # keep only real training runs (they have a configuration.yaml)
    run_dirs = [d for d in run_dirs if os.path.isfile(os.path.join(d, "configuration.yaml"))]

    if EventAccumulator is None:
        print("Warning: tensorboard not installed -> no progression.csv, "
              "rewards taken from timers.json")

    done = existing_run_ids(args.out)
    hardware = get_hardware()
    runs, progression = [], []

    for run_dir in run_dirs:
        run_id = os.path.basename(os.path.normpath(run_dir))
        if run_id in done:
            print(f"skip   {run_id} (already in {args.out})")
            continue
        try:
            row, prog = extract_run(run_dir, hardware)
        except (OSError, KeyError, StopIteration) as e:
            print(f"error  {run_id}: {e!r}")
            continue
        runs.append(row)
        progression += prog
        done.add(run_id)  # the same folder passed twice is only added once
        print(f"added  {run_id}: {row['total_steps']} steps, "
              f"{row['time_elapsed_seconds']} s, reward {row['cumulative_reward_mean']}")

    # progression.csv is written first and runs.csv last: a run only counts as done
    # once it is in runs.csv, so if a write fails the run is extracted again next time.
    # Runs already in progression.csv are not written there twice.
    append_rows(args.progression, PROGRESSION_COLUMNS, progression,
                skip_run_ids=existing_run_ids(args.progression))
    append_rows(args.out, RUN_COLUMNS, runs)
    print(f"\n{len(runs)} new run(s) -> {args.out}")


if __name__ == "__main__":
    main()
