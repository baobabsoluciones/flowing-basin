"""
RL Experiment 11
This script trains agents with RL Zoo's optimal hyperparameters
for SAC, A2C and PPO, with and without normalization
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A1"]
generals = ["G0", "G1"]
observations = ["O231"]
rewards = ["R1", "R22"]
trainings = [f"T{norm_digit}00{algo_digit}" for norm_digit in ["1", "5", "6"] for algo_digit in ["0", "1", "2"]]

for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):

    if training == "T1000":
        # This corresponds to training "T1", which was already done before
        continue

    if general == "G0" and reward == "R1" and training == "T1001":
        # This agent was correctly trained in a previous execution of this script that didn't finish
        continue

    rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    rl.train()

# Note we will train 1(A) * 2(G) * 1(O) * 2(R) * 8(T) - 2 = 32 agents
# Approximate training time: 32 agents * 50 min/agent = 27 hours
