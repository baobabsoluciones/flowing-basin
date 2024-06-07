"""
baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

GENERAL_CONFIGS = ['G0', 'G01', 'G1', 'G2', 'G21', 'G3']

for config in GENERAL_CONFIGS:
    Baseline(general_config=config, solver='PSO-RBO', tuned_hyperparams=True).solve(num_replications=5)

# for config in GENERAL_CONFIGS:
#     Baseline(general_config=config, solver='PSO', tuned_hyperparams=True).solve(num_replications=5)
