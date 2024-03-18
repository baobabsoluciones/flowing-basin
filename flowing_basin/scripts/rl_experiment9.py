"""
RL Experiment 9
This script trains agents with
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A113"]
generals = ["G0", "G1"]
observations = ["O2", "O4"]
rewards = ["R1", "R22"]
trainings = ["T3X"]  # WE NEED TO ADD THOSE THAT INCREASE THE MODEL'S SIZE!

for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):
    # Train for double the time only in the real environment
    training = training.replace("X", "4" if general == "G0" else "")
    rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    rl.train()

# Note we will train 2(G) * 2(O) * 7(R) + 2(G) * 2(O) = 28 + 4 = 32 agents
