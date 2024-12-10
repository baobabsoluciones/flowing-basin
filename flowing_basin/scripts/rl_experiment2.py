"""
RL Experiment 2
This script trains agents with normal and randomized features to check if the agents make use of the observation
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import json

ACTION = ["A1"]
GENERAL = ["G0", "G1"]
OBSERVATION = ["O2", "O20", "O201"]
REWARD = ["R1"]
TRAINING = ["T1"]
# TRAINING = ["T12"]

agents = []

for action, general, obs, reward, training in product(ACTION, GENERAL, OBSERVATION, REWARD, TRAINING):
    agents.append(f"rl-{action}{general}{obs}{reward}{training}")
    # rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    # rl.collect_obs()
    # rl.train()

experiment = {
    "description": """RL Experiment 2
This script trains agents with normal and randomized features to check if the agents make use of the observation""",
    "agents": agents
}
with open('experiments/experiment2.json', 'w') as f:
    json.dump(experiment, f, indent=4)
