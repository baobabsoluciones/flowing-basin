"""
baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

# configs = ['G0', 'G1', 'G2', 'G3']
# for config in configs:
#     Baseline(general_config=config, solver='PSO').solve(num_replications=5)

# Baseline(solver="Heuristic", general_config='G9').solve()
Baseline(solver="rl-greedy", general_config='G9').solve()

# Baseline(general_config='G2', solver='PSO').tune(num_trials=100, num_replications=1)
# Baseline(general_config='G3', solver='PSO').tune(num_trials=100, num_replications=1)

# Baseline(general_config='G2', solver='PSO', tuned_hyperparams=True).solve(num_replications=1)
# Baseline(general_config='G3', solver='PSO', tuned_hyperparams=True).solve(num_replications=1)

# Baseline(general_config='G1', solver='PSO-RBO', max_time=30).tune(
#     num_trials=4, instance_names=['Percentile25'], num_replications=1
# )
