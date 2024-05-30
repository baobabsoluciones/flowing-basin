"""
baseline.py
Script to compute solutions using PSO, MILP, etc. as baselines for RL
"""

from flowing_basin.solvers import Baseline

# configs = ['G0', 'G1', 'G2', 'G3']
# for config in configs:
#     Baseline(general_config=config, solver='PSO').solve(num_replications=5)

Baseline(solver="MILP", general_config='G01').solve()
Baseline(solver="MILP", general_config='G21').solve()

# Baseline(general_config='G01', solver='PSO').tune(num_trials=100, num_replications=1)
# Baseline(general_config='G21', solver='PSO').tune(num_trials=100, num_replications=1)

Baseline(general_config='G01', solver='PSO', tuned_hyperparams=True).solve(num_replications=1)
Baseline(general_config='G21', solver='PSO', tuned_hyperparams=True).solve(num_replications=1)
# Baseline(general_config='G01', solver='PSO', tuned_hyperparams=True, max_time=5).solve(num_replications=1, instance_names=['Percentile25'])
# Baseline(general_config='G21', solver='PSO', tuned_hyperparams=True, max_time=5).solve(num_replications=1, instance_names=['Percentile25'])

# Baseline(general_config='G1', solver='PSO-RBO', max_time=30).tune(
#     num_trials=4, instance_names=['Percentile25'], num_replications=1
# )
