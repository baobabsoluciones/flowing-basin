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


paths = get_experiment9_model_paths()
new_folder = "C:/Users/rodri/Documents/experiment9"
copy_folders(paths, new_folder)
