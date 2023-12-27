from flowing_basin.core import TrainingData, Solution
import matplotlib.pyplot as plt

MODEL_FILENAMES = [
    "2023-11-16 05.25_f=MLP_p=PCA",
    "2023-11-16 03.52_f=MLP_p=identity",
    "2023-11-16 06.55_f=CNN_p=identity"
]
BASELINES = [f"instancePercentile{percentile:>02}_LPmodel_k=2_PowerPenalties" for percentile in range(0, 101, 10)]

# Combine agents
training_objects = []
for model_filename in MODEL_FILENAMES:
    path_training_data = f"../solutions/rl_models/RL_model_{model_filename}/training_data.json"
    path_evaluation_data = f"../solutions/rl_models/RL_model_{model_filename}/evaluations.npz"
    training_object = TrainingData.from_json(path_training_data)
    training_check = training_object.check()
    if training_check:
        raise ValueError(f"Problems with the data: {training_check}")
    training_object.add_random_instances(model_filename.split("_")[0], path_evaluation_data)
    training_objects.append(training_object)
training = sum(training_objects)

# Add baselines
for baseline in BASELINES:
    path_sol = f"../solutions/rl_baselines/{baseline}.json"
    sol_object = Solution.from_json(path_sol)
    training += sol_object
print(training.data)

fig1, ax = plt.subplots()
training.plot_training_curves(ax, ['income', 'acc_reward'], 'fixed')
plt.show()

fig2, ax = plt.subplots()
training.plot_training_curves(ax, ['income', 'acc_reward'], ['Percentile50'])
plt.show()

# fig3, ax = plt.subplots()
# training.plot_training_curves(ax, ['acc_reward'], ['random'])
# plt.show()
# TODO: fix the ZeroDivisionError here
