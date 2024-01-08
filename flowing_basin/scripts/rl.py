"""
This script allows training a single RL agent
or analyzing the trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("A1G1O20R1T12", verbose=2)
# rl.check_train_env(max_timestep=4, obs_types=['raw', 'normalized', 'projected'], initial_date='2021-03-11 00:15')
# rl.collect_obs()
# rl.plot_histograms_projector_obs()
rl.train(save_agent=False)
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve()
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])

# regex = ".*G1O2(0\d?)?R1T1"  # Experiment 2
# ReinforcementLearning.plot_all_training_curves(regex)
# ReinforcementLearning.barchart_training_times()
# print("Average training time:", ReinforcementLearning.get_avg_training_time())
# ReinforcementLearning.print_max_avg_incomes(regex, permutation='GTOAR')
