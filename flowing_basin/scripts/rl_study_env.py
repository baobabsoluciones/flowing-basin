"""
RL Study Environment

This script allows studying a single agent.
"""

# Disable `tensorboard` when we do not intend to save the agent,
# because Tensorboard always raises an error on debug mode
SAVE_AGENT = False
if not SAVE_AGENT:
    import sys
    sys.modules['torch.utils.tensorboard'] = None
from flowing_basin.solvers.rl import ReinforcementLearning
import time

rl = ReinforcementLearning("rl-A1G2O231R22T1", verbose=3)
# rl.create_train_env()
# print("Configuration:", rl.config.to_dict())
rl.check_train_env(obs_types=['raw'], initial_date='2020-08-19 00:00', seed=42)  # instancePercentile50
# rl.collect_obs()
# rl.plot_histograms_projector_obs(show_lookback=False, show_projected=False)
# rl.train(save_agent=SAVE_AGENT)
# rl.plot_histograms_agent_obs()
# rl.plot_training_curve_agent(instances=["Percentile50"])
# rl.plot_training_curves_compare(["rl-A1G0O22R1T02", "rl-A1G0O221R1T02"], ["MILP"], values=["income", "acc_reward"])
# print(rl.run_agent("Percentile50").solution.to_dict())
# start = time.perf_counter()
# print(rl.run_agent([f"Percentile{i*10:02}" for i in range(0, 11)]))
# print("Time:", time.perf_counter() - start)
# ReinforcementLearning.print_max_avg_incomes(rl.agent_name)
