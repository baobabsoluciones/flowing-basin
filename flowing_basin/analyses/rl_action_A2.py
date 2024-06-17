"""
This script analyzes all the solutions with trained A2 agents
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from matplotlib import pyplot as plt
import math


PLOT_REWARD_TIMES_PRICE = False
SOLVER_NAMES = {'rl-greedy': 'Greedy'}
AGENT = "rl-A21G3O3R1T74"
AGENT_NAME = "RL agent"
INSTANCES = ["Percentile00", "Percentile10"]
FILENAME = f"rl_action_A2/adjustments_{AGENT}{'_' + '_'.join(INSTANCES) if INSTANCES is not None else ''}"

baseline_colors = {
    'MILP': 'black',
    'rl-greedy': 'gray'
}
rl = ReinforcementLearning(AGENT, verbose=2)
agent_name = AGENT_NAME if AGENT_NAME is not None else rl.agent_name
instances = rl.get_all_fixed_instances(rl.config.num_dams)
if INSTANCES is not None:
    instances = [instance for instance in instances if instance.get_instance_name() in INSTANCES]
num_cols = math.ceil(math.sqrt(len(instances)))
num_rows = math.ceil(len(instances) / num_cols)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
i = 0
for row in range(num_rows):
    for col in range(num_cols):
        if i < len(instances):
            instance = instances[i]
            run = rl.run_agent(instance)
            incomes = [sol.get_objective_function() for sol in run.solutions]
            total_rewards = [total_reward * instance.get_largest_price() for total_reward in run.total_rewards]
            if INSTANCES is None or len(INSTANCES) > 1:
                try:
                    ax = axs[row, col]
                except IndexError:
                    ax = axs[col]
            else:
                ax = axs
            label = agent_name
            if PLOT_REWARD_TIMES_PRICE:
                label += " (incomes)"
            ax.plot(incomes, marker='o', label=label)
            if PLOT_REWARD_TIMES_PRICE:
                ax.plot(total_rewards, marker='o', label=rl.agent_name + " (reward × price)")
            for baseline in rl.get_all_baselines(rl.config_names['G']):
                solver = baseline.get_solver()
                if baseline.get_instance_name() == instance.get_instance_name() and solver in baseline_colors.keys():
                    solver_name = SOLVER_NAMES[solver] if solver in SOLVER_NAMES else solver
                    ax.axhline(
                        y=baseline.get_objective_function(), color=baseline_colors[solver], linestyle='-',
                        label=solver_name
                    )
            # ax.set_xlabel('Steps')
            ax.set_xticks([])
            ax.set_ylabel('Income (€)')
            if INSTANCES is None:
                ax.set_title(instance.get_instance_name())
            if i == 0:
                fig.legend(loc='upper right')
        i += 1
fig.suptitle("Adjustments of " + agent_name + " over Greedy")
plt.tight_layout()
plt.savefig(FILENAME + '.eps')
plt.savefig(FILENAME + '.png')
plt.show()
