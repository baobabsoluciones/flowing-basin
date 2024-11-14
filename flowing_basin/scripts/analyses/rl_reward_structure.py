"""
RL Reward Structure
This script analyses the relationship between the average reward of rl-greedy
and the average reward of MILP and rl-random
"""

from flowing_basin.core import TrainingData, Instance
from flowing_basin.solvers.rl import ReinforcementLearning
import numpy as np
import matplotlib.pyplot as plt

GENERAL_CONFIG = 'G0'
BASELINE_COLORS = {
    'MILP': 'blue',
    'rl-random': 'red',
    'rl-greedy': 'grey'
}
REWARD_NAMES = {'R1': 'unadjusted reward', 'R22': 'Greedy-adjusted reward', 'R23': 'MILP-adjusted reward'}
STUDY_INCOMES = False  # Study incomes instead of rewards
PUT_CONFIG_TITLE = False
REWARD = 'R23'
AGENT = f"rl-A1{GENERAL_CONFIG}O232{REWARD}T1"  # Agent used to calculate rewards
FILENAME = f"rl_reward_structure/reward_{REWARD}"
MIN_BOTTOM = -10

if STUDY_INCOMES:

    # Get dict[baseline, dict[instance, income]] using TrainingData class
    y_axis = "Income"
    x_axis = "Income"
    training_data_baselines = TrainingData.create_empty()
    for baseline in ReinforcementLearning.get_all_baselines(GENERAL_CONFIG):
        training_data_baselines += baseline
    baseline_instances_values = training_data_baselines.get_baseline_instances_values()
    greedy_instances_values = baseline_instances_values['rl-greedy']

else:

    # Get dict[baseline, dict[instance, avg_reward]] using imitator agent
    # The imitator agent for the solvers will use the reward of the agent defined above,
    # while the imitator agent for rl-greedy will always use the R1 reward
    rl_for_solver = ReinforcementLearning(AGENT)
    reward_name = rl_for_solver.config_names['R']
    reward_name = REWARD_NAMES[reward_name] if reward_name in REWARD_NAMES else f"reward {reward_name}"
    y_axis = f"Average {reward_name}"
    baseline_instances_values = dict()

    for baseline in ReinforcementLearning.get_all_baselines(GENERAL_CONFIG):

        # Get the reward per timestep
        run = rl_for_solver.run_imitator(
            solution=baseline,
            instance=Instance.from_name(baseline.get_instance_name(), num_dams=rl_for_solver.config.num_dams)
        )
        avg_reward = sum(run.rewards_per_period) / len(run.rewards_per_period)

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
    x_axis = f"Average {REWARD_NAMES['R1']}"
    greedy_instances_values = dict()
    for baseline in ReinforcementLearning.get_all_baselines(GENERAL_CONFIG):
        if baseline.get_solver() == 'rl-greedy':
            run = rl_for_greedy.run_imitator(
                solution=baseline,
                instance=Instance.from_name(baseline.get_instance_name(), num_dams=rl_for_solver.config.num_dams)
            )
            avg_reward = sum(run.rewards_per_period) / len(run.rewards_per_period)
            greedy_instances_values[baseline.get_instance_name()] = avg_reward

print("Y values:", baseline_instances_values)
print("X values:", greedy_instances_values)

x_sorted_values = dict(sorted(greedy_instances_values.items()))
x = np.array(list(x_sorted_values.values()))
fig, ax = plt.subplots()
for solver, color in BASELINE_COLORS.items():
    y_sorted_values = dict(sorted(baseline_instances_values[solver].items()))
    y = np.array(list(y_sorted_values.values()))
    slope, intercept = np.polyfit(x, y, 1)
    print(f"Fitted line in {GENERAL_CONFIG} for {solver}: {slope} * x + {intercept} | R = {np.corrcoef(x, y)[0, 1]}")
    ax.scatter(x, y, color=color, label=solver)
    ax.plot(x, slope * x + intercept, color=color, linestyle='--')
    for i, instance_name in enumerate(y_sorted_values.keys()):
        ax.annotate(instance_name, (x[i], y[i]), color=color, xytext=(5, -5), textcoords='offset points')
current_bottom, current_top = ax.get_ylim()
if current_bottom < MIN_BOTTOM:
    ax.set_ylim(bottom=MIN_BOTTOM, top=current_top)
ax.set_xlabel(f'{x_axis} of Greedy')
ax.set_ylabel(f'{y_axis} of solver')
title = f'{y_axis} structure'
if PUT_CONFIG_TITLE:
    title += f" in {GENERAL_CONFIG}"
ax.set_title(title)
ax.legend()
ax.grid(True)
plt.savefig(FILENAME + '.eps')
plt.savefig(FILENAME + '.png')
plt.show()
