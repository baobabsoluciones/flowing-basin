"""
RL Study Experiments

This script allows analyzing several trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

regex = ".*"
agents = ReinforcementLearning.get_all_agents(regex)
print(f"Found {len(agents)} agents in folder:", agents)
# ReinforcementLearning.barchart_training_times(regex, hours=True)
ReinforcementLearning.print_training_times(regex, hours=True, csv_filepath="reports/training_times_experiments1-9.csv")
# ReinforcementLearning.print_max_avg_incomes(
#     regex, permutation='GOATR', baselines=["MILP", "rl-greedy", "rl-random"],
#     csv_filepath=f"reports/results_{general}_experiment9.csv"
# )
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.barchart_instances_incomes(regex, baselines=["rl-greedy"])
