"""
rl_baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

Baseline(general_config='G0', solver='MILP').solve(filename_tail="newpc")

# Baseline(general_config='G1', solver='PSO', max_time=5).tune(
#     num_trials=2, instance_names=['Percentile25'], num_replications=1
# )
# Baseline(general_config='G1', solver='PSO', max_time=5, tuned_hyperparams=True).solve(
#     instance_names=['Percentile25'],
# )

# Baseline(general_config='G1', solver='PSO', max_time=5).solve(
#     instance_names=['Percentile25'], num_replications=2
# )
