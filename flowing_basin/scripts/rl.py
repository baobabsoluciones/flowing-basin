"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

# rl = ReinforcementLearning("A1G0O1R1T11", verbose=2)
# rl.check_train_env(obs_types=['raw'], initial_date='2021-03-11 00:15')
# rl.collect_obs()
# rl.plot_histograms_projector_obs()
# rl.train()
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").to_dict())

regex = r'rl-A113G1O2R1T4$'
# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "Heuristic", "rl-greedy", "rl-random"])
# ReinforcementLearning.barchart_training_times(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.print_max_avg_incomes(regex, permutation='GTOAR')
ReinforcementLearning.barchart_instances_incomes(regex)
