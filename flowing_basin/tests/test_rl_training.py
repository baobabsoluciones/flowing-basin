from flowing_basin.core import Training
import matplotlib.pyplot as plt

MODEL_FILENAMES = [
    "2023-11-16 05.25_f=MLP_p=PCA",
    "2023-11-16 03.52_f=MLP_p=identity",
    "2023-11-16 06.55_f=CNN_p=identity"
]

training_objects = []
for model_filename in MODEL_FILENAMES:
    path_training_data = f"../solutions/rl_models/RL_model_{model_filename}/training.json"
    path_evaluation_data = f"../solutions/rl_models/RL_model_{model_filename}/evaluations.npz"
    training_object = Training.from_json(path_training_data)
    training_check = training_object.check()
    if training_check:
        raise ValueError(f"Problems with the data: {training_check}")
    training_object.add_random_instances(model_filename.split("_")[0], path_evaluation_data)
    training_objects.append(training_object)
training = sum(training_objects)

fig, axs = plt.subplots(1, 2)
training.plot_training_curves(axs[0], ['income', 'acc_reward'], 'fixed')
# training.plot_training_curves(axs[0], ['income'], 'fixed')
training.plot_training_curves(axs[1], ['acc_reward'], 'random')
plt.show()
