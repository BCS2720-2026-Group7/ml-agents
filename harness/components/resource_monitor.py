import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

SAMPLE_COLUMNS = [
    "timestamp",
    "elapsed_seconds",
    "process_count",        # processes in the monitored tree (so the root and its children)
    "process_cpu_percent",  # CPU of the tree (can exceed 100)
    "process_ram_mb",       # resident memory
    "system_cpu_percent",
    "system_ram_percent",
]

DEFAULT_INTERVAL = 1.0 # We can change this later but I think 1 second is good
_MB = 1024 ** 2

# This is the entrypoint
def monitor(pid, output_path, interval=DEFAULT_INTERVAL, include_children=True, append=False):
    if interval <= 0:
        raise ValueError("interval must be positive")
    root = psutil.Process(pid)
    folder = os.path.dirname(output_path) or "."

    procs = track_tree(root, {}, include_children)
    psutil.cpu_percent(interval=None)  # the system-wide reading needs priming too
    start = time.monotonic()
    next_tick = start + interval
    pending = []        # rows not yet written because the folder doesn't exist yet
    first_write = True
    count = 0

    try:
        while True:
            # sleep until the next tick on a fixed schedule, so samples don't drift
            time.sleep(max(0.0, next_tick - time.monotonic()))
            next_tick += interval
            if not is_alive(root):
                break
            procs = track_tree(root, procs, include_children)
            pending.append(take_sample(procs, start))
            count += 1
            if os.path.isdir(folder):  # wait for mlagents-learn to create the folder
                write_rows(output_path, pending, first_write, append)
                pending, first_write = [], False
    finally:
        if pending or first_write:
            os.makedirs(folder, exist_ok=True)
            write_rows(output_path, pending, first_write, append)
    return count

def track_tree(root, procs, include_children=True):
    current = [root]
    if include_children:
        try:
            current += root.children(recursive=True)
        except psutil.Error:
            pass
    tree = {}
    for proc in current:
        if proc.pid not in procs:
            try:
                proc.cpu_percent(interval=None)  # first call always returns 0: starts the measurement
            except psutil.Error:
                pass
        tree[proc.pid] = procs.get(proc.pid, proc)
    return tree

def take_sample(procs, start):
    # Measure the process tree and the whole machine once. Returns one CSV row as a dict.
    cpu = ram = 0.0
    count = 0
    for proc in procs.values():
        try:
            with proc.oneshot():
                cpu += proc.cpu_percent(interval=None)
                ram += proc.memory_info().rss
            count += 1
        except psutil.Error:
            continue  # exited since it was listed, or not ours to inspect

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "process_count": count,
        "process_cpu_percent": round(cpu, 1),
        "process_ram_mb": round(ram / _MB, 1),
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_ram_percent": psutil.virtual_memory().percent,
    }

def is_alive(proc):
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False

def write_rows(path, rows, first_write, append):
    """Add rows to the CSV. The first write starts a new file, unless `append` (e.g. --resume)."""
    existing = os.path.isfile(path) and os.path.getsize(path) > 0
    mode = "a" if (append or not first_write) else "w"
    with open(path, mode, newline="") as f:  # reopened every time, so a killed run loses nothing
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        if first_write and not (append and existing):
            writer.writeheader()
        writer.writerows(rows)

SUMMARY_COLUMNS = [
    "resource_samples",
    "process_cpu_percent_mean", "process_cpu_percent_max",
    "process_ram_mb_mean", "process_ram_mb_max",
    "system_cpu_percent_mean", "system_ram_percent_max",
]


def summarize_resource_usage(csv_path):
    summary = dict.fromkeys(SUMMARY_COLUMNS)
    summary["resource_samples"] = 0
    if not os.path.isfile(csv_path):
        return summary

    columns = {c: [] for c in ("process_cpu_percent", "process_ram_mb",
                               "system_cpu_percent", "system_ram_percent")}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                values = {c: float(row[c]) for c in columns}
            except (KeyError, TypeError, ValueError):
                continue  # skip a broken line (e.g. training was killed mid-write)
            for c, v in values.items():
                columns[c].append(v)

    n = len(columns["process_cpu_percent"])
    if n == 0:
        return summary

    def mean(values):
        return round(sum(values) / len(values), 2)

    summary.update({
        "resource_samples": n,
        "process_cpu_percent_mean": mean(columns["process_cpu_percent"]),
        "process_cpu_percent_max": max(columns["process_cpu_percent"]),
        "process_ram_mb_mean": mean(columns["process_ram_mb"]),
        "process_ram_mb_max": max(columns["process_ram_mb"]),
        "system_cpu_percent_mean": mean(columns["system_cpu_percent"]),
        "system_ram_percent_max": max(columns["system_ram_percent"]),
    })
    return summary

# CLI STUFF BELOW
def default_output_path(run_id, results_dir="harness/hand-off"):
    return os.path.join(results_dir, run_id, "run_logs", "resource_usage.csv")


def find_run_id(command):
    """The value of --run-id in an mlagents-learn command (both `--run-id X` and `--run-id=X`)."""
    for i, arg in enumerate(command):
        if arg.startswith("--run-id="):
            return arg.split("=", 1)[1]
        if arg == "--run-id" and i + 1 < len(command):
            return command[i + 1]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample CPU/RAM usage of a training run with psutil.",
        usage="%(prog)s [options] -- mlagents-learn CONFIG --run-id=RUN_ID [...]\n"
              "       %(prog)s [options] --pid PID --out FILE")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"seconds between samples (default {DEFAULT_INTERVAL})")
    parser.add_argument("--pid", type=int, help="monitor an already running process instead")
    parser.add_argument("--out", help="CSV file to write (default: results/<run_id>/run_logs/resource_usage.csv)")
    parser.add_argument("--results-dir", default="results",
                        help="results folder used to build the default --out (default: results)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="training command to launch and monitor, after `--`")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if bool(command) == bool(args.pid):
        parser.error("give either a command after `--` or --pid, not both")

    out = args.out
    if out is None:
        run_id = find_run_id(command)
        if run_id is None:
            parser.error("--out is required when the command has no --run-id")
        out = default_output_path(run_id, args.results_dir)

    if args.pid:
        print(f"monitoring pid {args.pid} -> {out} (Ctrl+C to stop)")
        try:
            count = monitor(args.pid, out, args.interval)
            print(f"[resource_monitor] process exited, {count} samples written to {out}")
        except KeyboardInterrupt:
            print(f"[resource_monitor] stopped, samples written to {out}")
        return 0

    process = subprocess.Popen(command)
    try:
        count = monitor(process.pid, out, args.interval, append="--resume" in command)
        print(f"[resource_monitor] {count} samples written to {out}", file=sys.stderr)
    except KeyboardInterrupt:
        # Ctrl+C reaches mlagents-learn too; wait below while it saves its model.
        print(f"[resource_monitor] stopped, samples written to {out}", file=sys.stderr)
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
