"""
rl_baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

# configs = ['G0', 'G1', 'G2', 'G3']
# for config in configs:
#     Baseline(general_config=config, solver='PSO').solve(num_replications=5)

# Baseline(general_config='G1', solver='PSO', max_time=5).tune(
#     num_trials=2, instance_names=['Percentile25'], num_replications=1
# )
Baseline(general_config='G1', solver='PSO-RBO', max_time=10).tune(
    num_trials=50, instance_names=['Percentile25'], num_replications=1
)

# Baseline(general_config='G1', solver='PSO', max_time=5, tuned_hyperparams=True).solve(
#     instance_names=['Percentile25'],
# )
