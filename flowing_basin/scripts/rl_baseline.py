"""
rl_baseline.py
TODO: once the Baseline class is finished, substitute lp_rl_baseline.py and heuristic_rl_baseline.py with this
"""

from flowing_basin.solvers import Baseline

print(Baseline(general_config='G0', solver='MILP').config)
print(Baseline(general_config='G0', solver='PSO').config)
print(Baseline(general_config='G0', solver='PSO-RBO').config)
print(Baseline(general_config='G0', solver='Heuristic').config)
print(Baseline(general_config='G1', solver='MILP').config)
print(Baseline(general_config='G1', solver='PSO').config)
print(Baseline(general_config='G1', solver='PSO-RBO').config)
print(Baseline(general_config='G1', solver='Heuristic').config)
