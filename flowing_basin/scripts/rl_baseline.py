"""
rl_baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

# print(Baseline(general_config='G0', solver='MILP').config)
# print(Baseline(general_config='G0', solver='PSO').config)
# print(Baseline(general_config='G0', solver='PSO-RBO').config)
# print(Baseline(general_config='G0', solver='Heuristic').config)
# print(Baseline(general_config='G1', solver='MILP').config)
# print(Baseline(general_config='G1', solver='PSO').config)
# print(Baseline(general_config='G1', solver='PSO-RBO').config)
# print(Baseline(general_config='G1', solver='Heuristic').config)

Baseline(general_config='G1', solver='PSO').solve()
# Baseline(general_config='G1', solver='MILP').solve()
