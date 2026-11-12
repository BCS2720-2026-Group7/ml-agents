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
import threading
import time
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


RUN_COLUMNS = [
    "run_id", "timestamp", "environment", "drl_algorithm", "seed",
    "cpu_model", "cpu_cores", "cpu_threads", "cpu_frequency_mhz", "total_ram_gb",
    "gpu_model", "operating_system", "torch_version", "mlagents_version",
    "batch_size", "buffer_size", "learning_rate", "learning_rate_schedule",
    "hidden_units", "num_layers", "normalize", "gamma", "max_steps", "time_horizon",
    "beta", "epsilon", "lambda", "num_epoch",
    "tau", "steps_per_update", "init_entcoef", "buffer_init_steps",
    "total_steps", "time_elapsed_seconds",
    "cumulative_reward_mean", "max_reward", "mean_reward_last3",
    "peak_ram_mb", "avg_cpu_percent"
]

PROGRESSION_COLUMNS = ["run_id", "step", "cumulative_reward", "episode_length",
                       "entropy", "learning_rate"]

TB_TAGS = {
    "Environment/Cumulative Reward": "cumulative_reward",
    "Environment/Episode Length": "episode_length",
    "Policy/Entropy": "entropy",
    "Policy/Learning Rate": "learning_rate",
}


def monitor_resources(stop_event, result_dict):
    if psutil is None:
        return
    peak_ram = 0
    cpu_usages = []
    while not stop_event.is_set():
        try:
            current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram:
                peak_ram = current_ram
            cpu_usages.append(psutil.cpu_percent(interval=0.5))
        except Exception:
            pass
    result_dict["peak_ram_mb"] = round(peak_ram, 2)
    result_dict["avg_cpu_percent"] = round(sum(cpu_usages) / len(cpu_usages), 2) if cpu_usages else 0.0


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
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:
        return "none/unknown"


def get_os_name():
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


def extract_run(run_dir, hardware):
    run_id = os.path.basename(os.path.normpath(run_dir))
    environment, hyperparameters = read_config(run_dir)

    resource_stats = {}
    if psutil is not None:
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, resource_stats))
        monitor_thread.start()
    else:
        resource_stats = {"peak_ram_mb": None, "avg_cpu_percent": None}

    row = {"run_id": run_id, "environment": environment}
    row.update(hardware)
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

    if psutil is not None:
        stop_event.set()
        monitor_thread.join()
        row.update(resource_stats)
    else:
        row.update(resource_stats)

    for p in progression:
        p["run_id"] = run_id
    return row, progression


def existing_run_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def drop_partial_last_line(path):
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def existing_progression_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if None in r.values():
                continue
            try:
                keys.add((r["run_id"], int(float(r["step"]))))
            except (KeyError, TypeError, ValueError):
                pass
    return keys


def append_rows(path, columns, rows):
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
    run_dirs = [d for d in run_dirs if os.path.isfile(os.path.join(d, "configuration.yaml"))]

    if EventAccumulator is None:
        print("Warning: tensorboard not installed -> no progression.csv, "
              "rewards taken from timers.json")

    for path in (args.out, args.progression):
        drop_partial_last_line(path)
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
        done.add(run_id)
        print(f"added  {run_id}: {row['total_steps']} steps, "
              f"{row['time_elapsed_seconds']} s, reward {row['cumulative_reward_mean']}")

    saved = existing_progression_keys(args.progression)
    progression = [p for p in progression if (p["run_id"], p["step"]) not in saved]
    append_rows(args.progression, PROGRESSION_COLUMNS, progression)
    append_rows(args.out, RUN_COLUMNS, runs)
    print(f"\n{len(runs)} new run(s) -> {args.out}")


if __name__ == "__main__":
    main()