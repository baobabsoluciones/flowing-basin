"""
baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

GENERAL_CONFIGS = ['G0', 'G01', 'G1', 'G2', 'G21', 'G3']

for config in GENERAL_CONFIGS:
    # Baseline(general_config=config, solver='PSO-RBO', max_time=30).solve(instance_names=['Percentile25'])
    Baseline(general_config=config, solver='PSO-RBO').solve(num_replications=5)

# for config in GENERAL_CONFIGS:
#     Baseline(general_config=config, solver='PSO-RBO', max_time=5).tune(num_trials=4, num_replications=1, instance_names=['Percentile25'])
#     # Baseline(general_config=config, solver='PSO-RBO').tune(num_trials=100, num_replications=1)
