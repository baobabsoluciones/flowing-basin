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

# Baseline(general_config='G3', solver='PSO', max_time=10).solve(instance_names=['Percentile00'])
# Baseline(general_config='G3', solver='MILP', max_time=40).solve(instance_names=['Percentile00'])

# Baseline(general_config='G1', solver='PSO', max_time=5).tune(
#     num_trials=2, instance_names=['Percentile25'], num_replications=1
# )
# Baseline(general_config='G1', solver='PSO', max_time=5, tuned_hyperparams=True).solve(
#     instance_names=['Percentile25'],
# )

Baseline(general_config='G1', solver='PSO', max_time=5).solve(
    instance_names=['Percentile25'], num_replications=2
)
