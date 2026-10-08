# Unity ML-Agents

Group 7 branch.

## Setup

Clone this repository. You must have SSH keys for git set up locally to be able to push.

```shell
git clone git@github.com:BCS2720-2026-Group7/ml-agents.git
```

### Python setup

These instructions have been tested as of 2026-10-06 at the latest, on a Linux machine.

Python 3.10.12 is the latest version confirmed compatible, so the installation instructions assume this version. All commands are run from the `ml-agents` directory.

Create a virtual environment.

```shell
python -m venv ./venv
```

Activate the virtual environment. The virtual environment is in `.gitignore`, so it does not get commited.

```shell
# UNIX
source venv/bin/activate
# Windows
venv\Scripts\activate
```

Check environment activation. This must print `Python 3.10.12`, anything else means an error in installation

```shell
python --version
```

Install packages.

```shell
python -m pip install -r requirements.txt
```

Check command installation

```shell
mlagents-learn --help
```

### Unity setup

These instructions apply to both UNIX and Windows machines.

Install Unity Hub from the official website and log into your account. Open the application, go to the Projects tab, select new project, and navigate to the location where the git clone was made. Select the `ml-agents/Project` directory. Select Unity Editor version `6000.0.77f1` and open the project. In the Unity Editor, navigate to the scenes directory.

```text
Project/Assets/ML-Agents/Examples/Soccer/Scenes
```

Open either `SoccerTwos.unity` or `StrikersVsGoalie.unity`. Start the simulation by clicking the Play button in the Unity GUI.

#### Connect environment to Python training

Run the training command. This command is run from the `ml-agents` directory with the virtual environment activated.

```shell
mlagents-learn config/ppo/SoccerTwos.yaml --run-id=soccer_test_01
```

Begin training. Press the Play button in the Unity GUI when the terminal prompts you to start.
