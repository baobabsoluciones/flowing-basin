"""
RL Reward Structure
This script analyses the relationship between the average reward of rl-greedy
and the average reward of MILP and rl-random
"""

from flowing_basin.core import TrainingData
from flowing_basin.solvers.rl import ReinforcementLearning
import numpy as np
import matplotlib.pyplot as plt

general_config = 'G0'
baseline_colors = {
    'MILP': 'blue',
    'rl-random': 'red'
}

training_data_baselines = TrainingData.create_empty()
for baseline in ReinforcementLearning.get_all_baselines(general_config):
    training_data_baselines += baseline
baseline_instances_values = training_data_baselines.get_baseline_instances_values()

sorted_values = dict(sorted(baseline_instances_values['rl-greedy'].items()))
x = np.array(list(sorted_values.values()))
fig, ax = plt.subplots()
for solver, color in baseline_colors.items():
    sorted_values = dict(sorted(baseline_instances_values[solver].items()))
    y = np.array(list(sorted_values.values()))
    slope, intercept = np.polyfit(x, y, 1)
    print(f"Fitted line in {general_config} for {solver}: {slope} * x + {intercept} | R = {np.corrcoef(x, y)[0, 1]}")
    ax.scatter(x, y, color=color, label=solver)
    ax.plot(x, slope * x + intercept, color=color, linestyle='--')
    for i, instance_name in enumerate(sorted_values.keys()):
        ax.annotate(instance_name, (x[i], y[i]), color=color, xytext=(5, -5), textcoords='offset points')
ax.set_xlabel('Income of rl-greedy')
ax.set_ylabel('Income of solver')
ax.set_title(f'Income structure in {general_config}')
ax.legend()
ax.grid(True)
plt.show()
