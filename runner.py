import os
import subprocess
import pathlib

from data_collection.resource_monitor import monitor, default_output_path

for i in os.scandir('./params'):
    if i.is_file():
        runID = pathlib.Path(i.name).stem
        print("Running", runID)
        cmd = ["mlagents-learn", i.path, "--env", "Project/Build/Worm/Worm.x86_64", "--no-graphics", "--run-id", runID, "--force"]
        process = subprocess.Popen(cmd)
        # Samples CPU/RAM of mlagents-learn and the Unity build every second
        # -> results/<runID>/run_logs/resource_usage.csv (returns when training ends)
        monitor(process.pid, default_output_path(runID))
        returnCode = process.wait()
        if returnCode == 0:
            # This needs to be tied in to the extraction function that puts data into CSV
            pass
        else:
            with open("errors.log", "a") as f:
                f.write(f"{runID} {returnCode}\n")
