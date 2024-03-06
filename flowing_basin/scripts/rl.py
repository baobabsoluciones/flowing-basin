"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A1G0O231R233T1", verbose=2)
rl.check_train_env(obs_types=['raw'], initial_date='2020-08-19 00:00')  # instancePercentile50
# rl.collect_obs()
# rl.plot_histograms_projector_obs(show_lookback=False, show_projected=False)
# rl.train()
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").to_dict())

# regex = [f"rl-A1G0O231R22T1"]  # rl-A113G1O232R22T3
# regex = [f"^(?=.*GX)(?!.*(?:A2|O1|T4|T2)).*$"]
# ReinforcementLearning.barchart_training_times(regex)
# regex = [agent.replace("X", "1") for agent in regex]
# ReinforcementLearning.print_max_avg_incomes(
#     regex, permutation='GOATR', baselines=["rl-greedy"], take_average=False
# )
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.barchart_instances_incomes(regex, baselines=["rl-greedy"])
