"""
RL Reward Structure
This script analyses the relationship between the average reward of rl-greedy
and the average reward of MILP and rl-random
"""

from flowing_basin.core import TrainingData, Instance
from flowing_basin.solvers.rl import ReinforcementLearning
import numpy as np
import matplotlib.pyplot as plt

general_config = 'G0'
baseline_colors = {
    'MILP': 'blue',
    'rl-random': 'red',
    'rl-greedy': 'grey'
}
study_incomes = False  # Study incomes instead of rewards
agent = f"rl-A1{general_config}O2R24T1"  # Agent used to calculate rewards

if study_incomes:

    # Get dict[baseline, dict[instance, income]] using TrainingData class
    y_axis = "Income"
    x_axis = "Income"
    training_data_baselines = TrainingData.create_empty()
    for baseline in ReinforcementLearning.get_all_baselines(general_config):
        training_data_baselines += baseline
    baseline_instances_values = training_data_baselines.get_baseline_instances_values()
    greedy_instances_values = baseline_instances_values['rl-greedy']

else:

    # Get dict[baseline, dict[instance, avg_reward]] using imitator agent
    # The imitator agent for the solvers will use the reward of the agent defined above,
    # while the imitator agent for rl-greedy will always use the R1 reward
    rl_for_solver = ReinforcementLearning(agent)
    y_axis = f"Average reward {rl_for_solver.config_names['R']}"
    baseline_instances_values = dict()

    for baseline in ReinforcementLearning.get_all_baselines(general_config):

        # Get the reward per timestep
        run = rl_for_solver.run_imitator(solution=baseline, instance=Instance.from_name(baseline.get_instance_name()))
        avg_reward = sum(run.rewards) / len(run.rewards)
        avg_reward /= rl_for_solver.config.num_actions_block

        # Add to results
        solver = baseline.get_solver()
        if baseline_instances_values.get(solver) is None:
            baseline_instances_values[solver] = dict()
        baseline_instances_values[solver][baseline.get_instance_name()] = avg_reward

    # Get rl-greedy's dict[instance, avg_reward]
    # The imitator for rl-greedy (x-axis) will use the R1 reward
    greedy_config_names = rl_for_solver.config_names
    greedy_config_names['R'] = 'R1'
    rl_for_greedy = ReinforcementLearning(''.join(greedy_config_names.values()))
    x_axis = f"Average reward R1"
    greedy_instances_values = dict()
    for baseline in ReinforcementLearning.get_all_baselines(general_config):
        if baseline.get_solver() == 'rl-greedy':
            run = rl_for_greedy.run_imitator(solution=baseline, instance=Instance.from_name(baseline.get_instance_name()))
            avg_reward = sum(run.rewards) / len(run.rewards)
            avg_reward /= rl_for_greedy.config.num_actions_block
            greedy_instances_values[baseline.get_instance_name()] = avg_reward

print("Y values:", baseline_instances_values)
print("X values:", greedy_instances_values)

x_sorted_values = dict(sorted(greedy_instances_values.items()))
x = np.array(list(x_sorted_values.values()))
fig, ax = plt.subplots()
for solver, color in baseline_colors.items():
    y_sorted_values = dict(sorted(baseline_instances_values[solver].items()))
    y = np.array(list(y_sorted_values.values()))
    slope, intercept = np.polyfit(x, y, 1)
    print(f"Fitted line in {general_config} for {solver}: {slope} * x + {intercept} | R = {np.corrcoef(x, y)[0, 1]}")
    ax.scatter(x, y, color=color, label=solver)
    ax.plot(x, slope * x + intercept, color=color, linestyle='--')
    for i, instance_name in enumerate(y_sorted_values.keys()):
        ax.annotate(instance_name, (x[i], y[i]), color=color, xytext=(5, -5), textcoords='offset points')
ax.set_xlabel(f'{x_axis} of rl-greedy')
ax.set_ylabel(f'{y_axis} of solver')
ax.set_title(f'{y_axis} structure in {general_config}')
ax.legend()
ax.grid(True)
plt.show()
