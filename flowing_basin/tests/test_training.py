from flowing_basin.core import Training
import matplotlib.pyplot as plt

MODEL_DATETIME = "2023-12-20 17.01_f=MLP_p=identity"
training_data = Training.from_json(f"../solutions/rl_models/RL_model_{MODEL_DATETIME}/training.json")

inconsistencies = training_data.check()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

# Plot with all instances
_, ax = plt.subplots()
training_data.plot_training_curves(ax, values=['income', 'acc_reward'], instances='fixed')
plt.show()

# Plot with just one instance
_, ax = plt.subplots()
training_data.plot_training_curves(ax, values=['income', 'acc_reward'], instances=["Percentile50"])
plt.show()
