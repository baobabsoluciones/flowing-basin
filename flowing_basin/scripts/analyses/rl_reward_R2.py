"""
This script analyzes the effect of reward type R2
"""

from flowing_basin.solvers.rl import ReinforcementLearning


def show_barcharts(rl_object: ReinforcementLearning):
    rl_object.barchart_instances_rewards(['R1', 'R21', 'R22'])
    rl_object.barchart_instances_rewards(['R1', 'R21', 'R22'], named_policy="greedy")
    rl_object.barchart_instances_rewards(['R1', 'R21', 'R22'], named_policy="random")


# Best agent in real environment
rl = ReinforcementLearning("rl-A1G0O2R1T1", verbose=2)
# rl.check_train_env(obs_types=[], initial_date='2021-03-11 00:15', seed=0)
show_barcharts(rl)

# Best agent in simplified environment
# rl = ReinforcementLearning("rl-A113G1O2R1T4", verbose=2)
# show_barcharts(rl)
