"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

# rl = ReinforcementLearning("A110G0O2R1T12", verbose=2)
# rl.check_train_env(obs_types=['raw'], initial_date='2021-03-11 00:15')
# rl.collect_obs()
# rl.plot_histograms_projector_obs()
# rl.train(save_agent=False)
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").to_dict())

regex = ".*A1(1\d)?G.O2R1.*(T1|T3)$"  # Experiment 3
# ReinforcementLearning.plot_all_training_curves(regex)
ReinforcementLearning.barchart_training_times(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.print_max_avg_incomes(regex, permutation='GTOAR')
