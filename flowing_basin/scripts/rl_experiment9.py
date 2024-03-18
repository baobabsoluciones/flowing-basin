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
trainings = ["T3X", "T3X1", "T3X2", "T3X3"]
# "T3X4", "T3X5", "T3X6" will not be tested since this experiment would take too long
# and there is little reason to scale only the actor and not the critic,
# given that both are affected when the action and observation spaces are bigger

for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):

    # Train for double the time only in the real environment
    training = training.replace("X", "4" if general == "G0" else "0")
    if training == "T30":
        training = "T3"

    # Avoid repeating the training of known agents
    if observation == "O2" and training == "T3":
        continue

    # Note we WILL "repeat" the training of T34 agents with O2 observation,
    # which are the same agents as those trained with T3 and O2, but with a longer training time

    rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    rl.train()

# Note we will train 1(A) * 2(G) * 2(O) * 2(R) * 4(T) - 2 = 30 agents
# Approximate training time: 30 agents * 7.5 hours/agent = 225 hours = 9.375 days
# Half the agents will take 5 hours (T30) and half will take 10 hours (T34), so we consider 7.5 hours/agent
