import os
import subprocess
import pathlib
import sys
import concurrent.futures
import queue
import shutil

if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <envCount>")

envCount = int(sys.argv[1])

INPUT_DIR = os.path.join("harness", "params", "worm")
TEMP_DIR = os.path.join("harness", "hand-off", "temp")
ERRORS_DIR = os.path.join("harness", "errors")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)

HARD_CAP = 6 
parallelCounts = min(HARD_CAP, max(1, os.cpu_count() - 2))

portQueue = queue.Queue()
for i in range(30000, 30000 + (parallelCounts * envCount), envCount):
    portQueue.put(i)

active_processes = []

def run(entry):
    if entry.is_file():
        runID = pathlib.Path(entry.name).stem
        print(f"Running {runID}")

        temp_yaml = os.path.join(TEMP_DIR, entry.name)
        shutil.copy(entry.path, temp_yaml) # Copy yaml used in temp folder

        port = portQueue.get()
        try:
            cmd = [
                "mlagents-learn", temp_yaml, 
                "--env", "Project/Build/Worm/Worm.x86_64", 
                "--no-graphics", "--run-id", runID, "--force", 
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
                    if os.path.exists(entry.path):
                        os.remove(entry.path)
                    print(f"Successfully finished and cleaned up {runID}")
                else:
                    with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                        f.write(f"Extraction failed with exit code {extract_proc.returncode}\n")
            else:
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)
                with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                    f.write(f"Training failed with exit code {returnCode}\n")
        except Exception as e:
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
            with open(os.path.join(ERRORS_DIR, f"{runID}.txt"), "w") as f:
                f.write(f"Exception generated: {e}\n")
        finally:
            portQueue.put(port)

# Adapted from https://docs.python.org/3.10/library/concurrent.futures.html#threadpoolexecutor-example
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelCounts) as executor:
        entries = []
        if os.path.exists(INPUT_DIR):
            for i in os.scandir(INPUT_DIR):
                if i.name.endswith(".yaml"):
                    runID = pathlib.Path(i.name).stem
                    error_log_path = os.path.join(ERRORS_DIR, f"{runID}.txt")
                    
                    if not os.path.exists(error_log_path):
                        entries.append(i)

        entries.sort(key=lambda x: int(x.name.split('-')[0]) if '-' in x.name else 0)

        futureToParameters = {executor.submit(run, entry): entry for entry in entries}
        
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