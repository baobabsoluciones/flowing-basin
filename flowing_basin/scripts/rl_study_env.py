"""
RL Study Environment

This script allows studying a single agent.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("rl-A31G0O2R1T1302", verbose=2)
print("Configuration:", rl.config.to_dict())
rl.check_train_env(obs_types=['raw'], initial_date='2020-08-19 00:00', seed=42)  # instancePercentile50
# rl.collect_obs()
# rl.plot_histograms_projector_obs(show_lookback=False, show_projected=False)
# rl.train(save_agent=False)
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").solution.to_dict())
