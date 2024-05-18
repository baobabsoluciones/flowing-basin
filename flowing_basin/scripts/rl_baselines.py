"""
rl_baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baselines
import matplotlib.pyplot as plt

# Baselines(solvers=['MILP', 'PSO'], general_config='G0').barchart_instances()

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
for i, ax in enumerate(axes):
    Baselines(solvers=['MILP', 'PSO'], general_config=f'G{i}').barchart_instances_ax(ax)
plt.tight_layout()
plt.savefig("reports/barchart_instances_MILP_PSO.png")
plt.show()
