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
