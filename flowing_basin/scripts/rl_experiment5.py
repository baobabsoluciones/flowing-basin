"""
RL Experiment 5
This script trains agents with QuantilePseudoDiscretizer alone (O222),
or with a reward set using rl-greedy as a reference (R22).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import json

actions = ["A1", "A113"]
generals = ["G0", "G1"]

agents = []

for action, general in product(actions, generals):

    # Use a smaller replay buffer when action is A113, which has too big observation arrays
    training = "T1" if action != "A113" else "T3"
    # training = "T13" if action != "A113" else "T33"

    obs = "O222"
    reward = "R1"

    agents.append(f"rl-{action}{general}{obs}{reward}{training}")
    # rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    # rl.collect_obs()
    # rl.train()

    obs = "O2"
    reward = "R22"

    agents.append(f"rl-{action}{general}{obs}{reward}{training}")
    # rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    # rl.collect_obs()
    # rl.train()

experiment = {
    "description": """RL Experiment 5
This script trains agents with QuantilePseudoDiscretizer alone (O222),
or with a reward set using rl-greedy as a reference (R22).""",
    "agents": agents
}
with open('experiments/experiment5.json', 'w') as f:
    json.dump(experiment, f, indent=4)
