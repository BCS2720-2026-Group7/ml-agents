import os
import subprocess
import pathlib
import sys
import concurrent.futures
import queue
import shutil
import threading

if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <envCount>")

envCount = int(sys.argv[1])

INPUT_DIR = os.path.join("harness", "params", "worm")
TEMP_DIR = os.path.join("harness", "hand-off", "temp")
ERRORS_DIR = os.path.join("harness", "errors")
RESULTS_DIR = os.path.join("harness", "results")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)

# Once per seed
seeds = [0, 1, 2]

HARD_CAP = 6
parallelCounts = min(HARD_CAP, max(1, os.cpu_count() - 2))

portQueue = queue.Queue()
for i in range(30000, 30000 + (parallelCounts * envCount), envCount):
    portQueue.put(i)

active_processes = []

# Seeds remaining
seedsRemaining = {}
seedsLock = threading.Lock()

def run(entry, seed):
    if entry.is_file():
        runID = f"{pathlib.Path(entry.name).stem}-s{seed}"
        print(f"Running {runID}")

        temp_yaml = os.path.join(TEMP_DIR, f"{runID}.yaml")
        shutil.copy(entry.path, temp_yaml) # Copy yaml used in temp folder

        port = portQueue.get()
        try:
            cmd = [
                "mlagents-learn", temp_yaml,
                "--env", "Project/Build/Worm/Worm.x86_64",
                "--no-graphics", "--run-id", runID, "--force", "--seed", str(seed),
                "--num-envs", str(envCount), "--base-port", str(port),
                "--results-dir", TEMP_DIR
            ]
            process = subprocess.Popen(cmd)
            active_processes.append(process)
            returnCode = process.wait()

            if process in active_processes:
                active_processes.remove(process)

            temp_run_folder = os.path.join(TEMP_DIR, runID)

            if returnCode == 0:
                extract_cmd = [
                    "python", "-m", "harness.components.extract_run",
                    temp_run_folder
                ]
                extract_proc = subprocess.run(extract_cmd)

                if extract_proc.returncode == 0:
                    if os.path.exists(temp_yaml):
                        os.remove(temp_yaml)
                    if os.path.exists(temp_run_folder):
                        shutil.rmtree(temp_run_folder)
                    with seedsLock:
                        seedsRemaining[entry.path] -= 1
                        if seedsRemaining[entry.path] == 0 and os.path.exists(entry.path):
                            os.remove(entry.path)
                    print(f"Successfully finished and cleaned up {runID}")
                else:
                    print(f"{runID} extraction failed with exit code {extract_proc.returncode}")
                    with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                        f.write(f"Extraction failed with exit code {extract_proc.returncode}\n")
            else:
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)
                print(f"{runID} training failed with exit code {returnCode}")
                with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                    f.write(f"Training failed with exit code {returnCode}\n")
        except Exception as e:
            print(f"{runID} generated an exception: {e}")
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
            with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                f.write(f"Exception generated: {e}\n")
        finally:
            portQueue.put(port)

# Adapted from https://docs.python.org/3.10/library/concurrent.futures.html#threadpoolexecutor-example
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelCounts) as executor:
        jobs = []
        if os.path.exists(INPUT_DIR):
            for i in os.scandir(INPUT_DIR):
                if i.name.endswith(".yaml"):
                    seedsRemaining[i.path] = len(seeds)
                    for seed in seeds:
                        runID = f"{pathlib.Path(i.name).stem}-s{seed}"
                        error_log_path = os.path.join(ERRORS_DIR, f"{runID}.txt")
                        results_path = os.path.join(RESULTS_DIR, f"{runID}-results")

                        if os.path.exists(results_path):
                            seedsRemaining[i.path] -= 1
                        elif not os.path.exists(error_log_path):
                            jobs.append((i, seed))

        jobs.sort(key=lambda j: (seeds.index(j[1]), int(j[0].name.split('-')[0]) if '-' in j[0].name else 0))

        futureToParameters = {executor.submit(run, entry, seed): (entry, seed) for entry, seed in jobs}

        for future in concurrent.futures.as_completed(futureToParameters):
            future.result()

except KeyboardInterrupt:
    print("\n[!] Ctrl+C detected! Terminating all active trainings and shutting down runner...")
    for p in active_processes:
        try:
            p.terminate()
        except Exception:
            pass
    os._exit(1)
