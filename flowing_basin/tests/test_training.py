from flowing_basin.core import Training
import matplotlib.pyplot as plt

MODEL_DATETIME = "2023-11-09 01.48"
training_data = Training.from_json(f"../solutions/rl_models/RL_model_{MODEL_DATETIME}/training.json")

inconsistencies = training_data.check()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

# Plot with all instances
_, ax = plt.subplots()
training_data.plot_training_curves(ax)
plt.show()

# Plot with just one instance
_, ax = plt.subplots()
training_data.plot_training_curves(ax, ["Percentile50"])
plt.show()
