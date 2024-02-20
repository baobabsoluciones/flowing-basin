"""
This script analyzes all the solutions with trained A2 agents
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from matplotlib import pyplot as plt
import math


baseline_colors = {
    'MILP': 'black',
    'rl-greedy': 'gray'
}
rl = ReinforcementLearning("rl-A23G0O3R1T3", verbose=2)
instances = rl.get_all_fixed_instances()
num_cols = math.ceil(math.sqrt(len(instances)))
num_rows = math.ceil(len(instances) / num_cols)
fig, axs = plt.subplots(num_rows, num_cols)
i = 0
for row in range(num_rows):
    for col in range(num_cols):
        if i < len(instances):
            instance = instances[i]
            run = rl.run_agent(instance)
            incomes = [sol.get_objective_function() for sol in run.solutions]
            total_rewards = [total_reward * instance.get_largest_price() for total_reward in run.total_rewards]
            ax = axs[row, col]
            ax.plot(incomes, marker='o', label=rl.agent_name + " (incomes)")
            ax.plot(total_rewards, marker='o', label=rl.agent_name + " (reward × price)")
            for baseline in rl.get_all_baselines(rl.config_names['G']):
                solver = baseline.get_solver()
                if baseline.get_instance_name() == instance.get_instance_name() and solver in baseline_colors.keys():
                    ax.axhline(
                        y=baseline.get_objective_function(), color=baseline_colors[solver], linestyle='-', label=solver
                    )
            # ax.set_xlabel('Steps')
            ax.set_xticks([])
            ax.set_ylabel('Income (€)')
            ax.set_title(instance.get_instance_name())
            if i == 0:
                fig.legend(loc='upper right')
        i += 1
fig.suptitle("Adjustments of " + rl.agent_name)
plt.tight_layout()
plt.show()
