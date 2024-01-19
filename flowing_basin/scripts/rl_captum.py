"""
This script allows analyzing a trained agent
using captum.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

rl = ReinforcementLearning("rl-A1G0O2R1T1", verbose=2)
rl.integrated_gradients()
