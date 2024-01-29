"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A21G0O3R1T13", verbose=2)
rl.check_train_env(obs_types=['projected'], initial_date='2020-08-19 00:00')  # instancePercentile50
# rl.collect_obs()
# rl.plot_histograms_projector_obs()
# rl.train()
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").to_dict())

# regex = ["rl-A1G.O2R1T1", "rl-A1G.O2R22T1", "rl-A113G.O2R1T3", "rl-A113G.O2R22T3"]
# ReinforcementLearning.barchart_training_times(regex)
# regex = [agent.replace(".", "1") for agent in regex]
# ReinforcementLearning.print_max_avg_incomes(regex, permutation='GATOR', baselines=["MILP", "rl-greedy", "rl-random"])
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.barchart_instances_incomes(regex)
