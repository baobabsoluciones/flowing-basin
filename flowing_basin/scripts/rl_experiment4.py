"""
RL Experiment 4
This script trains different versions of the agent having 99 actions per block
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

# Common configuration
action = "A113"
reward = "R1"

ENVIRONMENTS = ["G0", "G1"]  # Real, simplified
OBSERVATIONS = ["O2", "O1"]  # Normal, more informative
TRAININGS = ["T3", "T4"]  # Normal, higher learning rate

for env, obs, training in product(ENVIRONMENTS, OBSERVATIONS, TRAININGS):

    # Do not train on simplified env with normal observation and learning rate
    # since we observed that the model already converges quickly in this case,
    # so continuing the training will not be beneficial
    if (env, obs, training) == ("G1", "O2", "T3"):
        continue

    rl = ReinforcementLearning(f"{action}{env}{obs}{reward}{training}", verbose=2)
    rl.train()

# In total 2 * 3 - 1 = 5 agents ==> 5 * 5 = 25 hours of training
