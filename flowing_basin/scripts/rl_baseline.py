"""
rl_baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

for config in ['G0', 'G1']:
    Baseline(general_config=config, solver='Heuristic').solve()

# Baseline(general_config='G1', solver='PSO', max_time=5).tune(
#     num_trials=2, instance_names=['Percentile25'], num_replications=1
# )
# Baseline(general_config='G1', solver='PSO', max_time=5, tuned_hyperparams=True).solve(
#     instance_names=['Percentile25'],
# )

# Baseline(general_config='G1', solver='PSO', max_time=30).solve(
#     instance_names=['Percentile25'], num_replications=1
# )
