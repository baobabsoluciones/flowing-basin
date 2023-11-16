from flowing_basin.core import Training
import matplotlib.pyplot as plt

MODEL_FILENAMES = [
    "2023-11-16 05.25_f=MLP_p=PCA",
    "2023-11-16 03.52_f=MLP_p=identity",
    "2023-11-16 06.55_f=CNN_p=identity"
]
MODEL_FILENAME = MODEL_FILENAMES[0]
path_training_data = f"../solutions/rl_models/RL_model_{MODEL_FILENAME}/training.json"
path_evaluation_data = f"../solutions/rl_models/RL_model_{MODEL_FILENAME}/evaluations.npz"
training = Training.from_json(path_training_data)
print(training.check())
training.add_random_instances(MODEL_FILENAME.split("_")[0], path_evaluation_data)

fig, axs = plt.subplots(1, 2)
training.plot_training_curves(axs[0], ['income', 'acc_reward'], 'fixed')
# training.remove_agent("random")
training.plot_training_curves(axs[1], ['acc_reward'], 'random')
plt.show()
