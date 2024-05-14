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

Baseline(general_config='G1', solver='PSO', max_time=5).tune(
    num_trials=2, instance_names=['Percentile25'], num_replications=1
)
Baseline(general_config='G1', solver='PSO', max_time=5, tuned_hyperparams=True).solve(
    instance_names=['Percentile25'],
)

# from flowing_basin.solvers import PSOConfiguration, PSO
# from flowing_basin.core import Instance
# config = PSOConfiguration(startups_penalty=0.0, limit_zones_penalty=0.0, volume_objectives={}, volume_shortage_penalty=0.0, volume_exceedance_bonus=0.0, flow_smoothing=0, num_particles=200, cognitive_coefficient=1.1619646770905345, social_coefficient=2.2003942087999007, inertia_weight=0.7041251591653152, use_relvars=True, max_relvar=1.0, bounds_handling='nearest', topology='pyramid', max_time=5, mode='linear')
# pso = PSO(config=config, instance=Instance.from_name('Percentile75', num_dams=2))
# pso.solve()
