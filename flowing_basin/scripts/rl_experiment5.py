"""
RL Experiment 5
This script trains agents with QuantilePseudoDiscretizer alone (O222),
or with a reward set using rl-greedy as a reference (R22).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A1", "A113"]
generals = ["G0", "G1"]

for action, general in product(actions, generals):

    # Use a smaller replay buffer when action is A113, which has too big observation arrays
    training = "T1" if action != "A113" else "T3"
    # training = "T11" if action != "A113" else "T31"

    obs = "O222"
    reward = "R1"

    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()

    obs = "O2"
    reward = "R22"

    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()
