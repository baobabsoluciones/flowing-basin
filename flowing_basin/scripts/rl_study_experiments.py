"""
RL Study Experiments

This script allows analyzing several trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product


def get_experiment9_agents(general) -> list[str]:
    actions = ["A113"]
    observations = ["O2", "O4"]
    rewards = ["R1", "R22"]
    trainings = ["T3X", "T3X1", "T3X2", "T3X3"]
    agents = []
    for action, observation, reward, training in product(actions, observations, rewards, trainings):
        training = training.replace("X", "4" if general == "G0" else "0")
        if training == "T30":
            training = "T3"
        if observation == "O2" and training == "T3":
            continue
        agent = f"rl-{action}{general}{observation}{reward}{training}"
        agents.append(agent)
    print(f"Returning {len(agents)} agents:", agents)
    return agents


regex = get_experiment9_agents('G0') + get_experiment9_agents('G1')
agents = ReinforcementLearning.get_all_agents(regex)
print(f"Found {len(agents)} agents in folder:", agents)
ReinforcementLearning.barchart_training_times(regex, hours=True)
# ReinforcementLearning.print_max_avg_incomes(
#     regex, permutation='GOATR', baselines=["MILP", "rl-greedy", "rl-random"],
#     csv_filepath=f"reports/results_{general}_experiment9.csv"
# )
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.barchart_instances_incomes(regex, baselines=["rl-greedy"])
