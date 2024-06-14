import os
import shutil
from itertools import product
from flowing_basin.solvers.rl import ReinforcementLearning


def copy_folders(paths: list[str], new_folder: str):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for path in paths:
        if os.path.exists(path) and os.path.isdir(path):
            folder_name = os.path.basename(path)
            destination_path = os.path.join(new_folder, folder_name)
            shutil.copytree(path, destination_path)
            print(f"Folder '{folder_name}' copied to '{new_folder}'")
        else:
            print(f"Skipping '{path}' as it either doesn't exist or is not a directory.")


def get_experiment9_model_paths() -> list[str]:
    actions = ["A113"]
    generals = ["G0", "G1"]
    observations = ["O2", "O4"]
    rewards = ["R1", "R22"]
    trainings = ["T3X", "T3X1", "T3X2", "T3X3"]
    paths = []
    for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):
        training = training.replace("X", "4" if general == "G0" else "0")
        if training == "T30":
            training = "T3"
        if observation == "O2" and training == "T3":
            continue
        agent = f"rl-{action}{general}{observation}{reward}{training}"
        paths.append(os.path.join(ReinforcementLearning.models_folder, agent))
    print(f"Returning {len(paths)} paths:", paths)
    return paths


def get_experiment11_model_paths() -> list[str]:
    actions = ["A1"]
    generals = ["G0", "G1"]
    observations = ["O231"]
    rewards = ["R1", "R22"]
    trainings = [f"T{norm_digit}00{algo_digit}" for norm_digit in ["1", "5", "6"] for algo_digit in ["0", "1", "2"]]
    paths = []
    for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):
        if training == "T1000":
            # This corresponds to training "T1", which was already done before
            continue
        agent = f"rl-{action}{general}{observation}{reward}{training}"
        paths.append(os.path.join(ReinforcementLearning.models_folder, agent))
    print(f"Returning {len(paths)} paths:", paths)
    return paths


def get_experiment12_model_paths() -> list[str]:
    agents = [
        "rl-A31G2O231R1T1402",
        "rl-A31G3O231R1T1402",
        "rl-A1G2O231R22T14",
        "rl-A1G3O231R22T14",
        "rl-A1G2O2R1T14",
        "rl-A1G3O2R1T14",
        "rl-A23G2O3R1T74",
        "rl-A23G2O3R1T748",
        "rl-A23G3O3R1T74",
        "rl-A23G3O3R1T748",
        "rl-A113G3O2R22T342",
        "rl-A113G3O2R22T749",
        "rl-A113G3O232R22T74",
        "rl-A113G3O232R22T748"
    ]
    paths = [os.path.join(ReinforcementLearning.models_folder, agent) for agent in agents]
    print(f"Returning {len(paths)} paths:", paths)
    return paths


def get_experiment12cont_model_paths() -> list[str]:
    agents = [
        "rl-A21G2O3R1T74",
        "rl-A21G2O3R1T748",
        "rl-A21G3O3R1T74",
        "rl-A21G3O3R1T748",
    ]
    paths = [os.path.join(ReinforcementLearning.models_folder, agent) for agent in agents]
    print(f"Returning {len(paths)} paths:", paths)
    return paths


if __name__ == "__main__":

    model_paths = get_experiment12cont_model_paths()
    new_folder = "C:/Users/rodrigo/Documents/experiment12cont"
    copy_folders(model_paths, new_folder)
