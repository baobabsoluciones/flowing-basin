"""
baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

GENERAL_CONFIGS = ['G0', 'G01', 'G1', 'G2', 'G21', 'G3']

# (general_config='G0', solver='rl-A31G0O231R1T1002').solve(num_replications=1)
Baseline(general_config='G1', solver='rl-A113G1O2R22T302').solve(num_replications=1)

# for config in GENERAL_CONFIGS:
#     Baseline(general_config=config, solver='PSO-RBO', tuned_hyperparams=True).solve(num_replications=5)

# for config in GENERAL_CONFIGS:
#     Baseline(general_config=config, solver='PSO', tuned_hyperparams=True).solve(num_replications=5)
