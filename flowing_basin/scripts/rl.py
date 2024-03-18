"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A113G0O2R1T346", verbose=2)
# rl.check_train_env(obs_types=['raw'], initial_date='2020-08-19 00:00', seed=42)  # instancePercentile50
# rl.collect_obs()
# rl.plot_histograms_projector_obs(show_lookback=False, show_projected=False)
rl.train(save_agent=False)
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").to_dict())

# regex = [f"rl-A1G0O231R22T1"]
# regex = [r'.*GX.*?(R22\d|R23\d?).*']
# agents = ReinforcementLearning.get_all_agents(regex)
# print(len(agents), agents)
# ReinforcementLearning.barchart_training_times(regex, hours=True)
# general = "G1"
# regex = [agent.replace("GX", general) for agent in regex]
# ReinforcementLearning.print_max_avg_incomes(
#     regex, permutation='GOATR', baselines=["MILP", "rl-greedy", "rl-random"],
#     csv_filepath=f"reports/results_{general}_experiment8.csv"
# )
# ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

# ReinforcementLearning.print_spaces(regex)
# ReinforcementLearning.print_training_times(regex)
# print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
# ReinforcementLearning.barchart_instances_incomes(regex, baselines=["rl-greedy"])
