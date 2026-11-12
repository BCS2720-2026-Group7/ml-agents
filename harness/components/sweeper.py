import os
import itertools
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS_DIR = os.path.join(BASE_DIR, "harness", "hand-off", "params")

def generate_ppo_sweep():
    template_path = os.path.join(BASE_DIR, "config", "ppo", "Worm.yaml")

    #get file
    if not os.path.exists(template_path):
        template_path = os.path.join(BASE_DIR, "Worm_ppo.yaml")

    with open(template_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Decision 1: most 5 influential for PPO
    grid = {
        "learning_rate": [0.0001, 0.0003, 0.001],
        "hidden_units": [256, 512],
        "num_layers": [2, 3],
        "gamma": [0.99, 0.995],
        "epsilon": [0.1, 0.2]
    }

    keys = grid.keys()
    values = grid.values()
    combinations = list(itertools.product(*values)) #gridsearch for all types

    print(f"Generating {len(combinations)} PPO configurations...") #ui for general knowledge

    for combo in combinations:
        params = dict(zip(keys, combo))

        config = yaml.safe_load(yaml.dump(base_config)) #copy base config

        behavior = config["behaviors"]["Worm"] #search file

        #update lerning rate & epsilon
        behavior["hyperparameters"]["learning_rate"] = params["learning_rate"] 
        behavior["hyperparameters"]["epsilon"] = params["epsilon"]

        #update gamma
        if "reward_signals" in behavior and "extrinsic" in behavior["reward_signals"]:
            behavior["reward_signals"]["extrinsic"]["gamma"] = params["gamma"]

        #update hidden units & num layers
        behavior["network_settings"]["hidden_units"] = params["hidden_units"]
        behavior["network_settings"]["num_layers"] = params["num_layers"]

        #custom filename
        run_name = (
            f"worm_ppo"
            f"_lr_{params['learning_rate']}"
            f"_hu_{params['hidden_units']}"
            f"_nl_{params['num_layers']}"
            f"_g_{params['gamma']}"
            f"_eps_{params['epsilon']}"
        )

        #push file
        out_path = os.path.join(PARAMS_DIR, f"{run_name}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Successfully generated PPO files in: {PARAMS_DIR}") #ui for general knowledge


def generate_sac_sweep():
    template_path = os.path.join(BASE_DIR, "config", "sac", "Worm.yaml")

    #get file
    if not os.path.exists(template_path):
        template_path = os.path.join(BASE_DIR, "Worm_sac.yaml")

    with open(template_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Decision 1: most 5 influential for SAC
    grid = {
        "learning_rate": [0.0001, 0.0003, 0.001],
        "hidden_units": [256, 512],
        "num_layers": [2, 3],
        "gamma": [0.99, 0.995],
        "buffer_size": [250000, 500000]
    }

    keys = grid.keys()
    values = grid.values()
    combinations = list(itertools.product(*values)) #gridsearch for all types

    print(f"Generating {len(combinations)} SAC configurations...") #ui for general knowledge

    for combo in combinations:
        params = dict(zip(keys, combo))

        config = yaml.safe_load(yaml.dump(base_config)) #copy base config

        behavior = config["behaviors"]["Worm"]#search file

        #update learning rate & buffer size
        behavior["hyperparameters"]["learning_rate"] = params["learning_rate"]
        behavior["hyperparameters"]["buffer_size"] = params["buffer_size"]

        #update gamma
        if "reward_signals" in behavior and "extrinsic" in behavior["reward_signals"]:
            behavior["reward_signals"]["extrinsic"]["gamma"] = params["gamma"]

        #update hidden units & num layers
        behavior["network_settings"]["hidden_units"] = params["hidden_units"]
        behavior["network_settings"]["num_layers"] = params["num_layers"]

        #custom filename
        run_name = (
            f"worm_sac"
            f"_lr_{params['learning_rate']}"
            f"_hu_{params['hidden_units']}"
            f"_nl_{params['num_layers']}"
            f"_g_{params['gamma']}"
            f"_bs_{params['buffer_size']}"
        )

        #push file
        out_path = os.path.join(PARAMS_DIR, f"{run_name}.yaml") 
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"Successfully generated SAC files in: {PARAMS_DIR}") #ui for general knowledge

if __name__ == "__main__":
    generate_sac_sweep()
    generate_ppo_sweep()