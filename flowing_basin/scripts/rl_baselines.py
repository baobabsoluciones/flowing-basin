"""
rl_baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baselines

Baselines(solvers=['MILP'], general_config='G0', include_folders=['old']).barchart_instances()
