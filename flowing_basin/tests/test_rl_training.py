from flowing_basin.core import Training
import matplotlib.pyplot as plt

MODEL_FILENAMES = [
    "2023-11-16 05.25_f=MLP_p=PCA",
    "2023-11-16 03.52_f=MLP_p=identity",
    "2023-11-16 06.55_f=CNN_p=identity",
    "2023-11-16 14.00_f=CNN_p=identity"
]
MODEL_FILENAME = MODEL_FILENAMES[-1]
path_training_data = f"../solutions/rl_models/RL_model_{MODEL_FILENAME}/training.json"
training = Training.from_json(path_training_data)

fig, ax = plt.subplots()
training.plot_training_curves(ax)
plt.show()
