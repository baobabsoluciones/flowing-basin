"""
RL Experiment 4
This script trains different versions of the agent having 99 actions per block
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import json

# Common configuration
action = "A113"
reward = "R1"

ENVIRONMENTS = ["G0", "G1"]  # Real, simplified
OBSERVATIONS = ["O2", "O1"]  # Normal, more informative
TRAININGS = ["T3", "T4"]  # Normal, higher learning rate
# TRAININGS = ["T31", "T41"]

agents = []

for env, obs, training in product(ENVIRONMENTS, OBSERVATIONS, TRAININGS):

    # Do not train on simplified env with normal observation and learning rate
    # since we observed that the model already converges quickly in this case,
    # so continuing the training will not be beneficial
    if (env, obs, training) == ("G1", "O2", "T3"):
        continue

    agents.append(f"rl-{action}{env}{obs}{reward}{training}")
    # rl = ReinforcementLearning(f"{action}{env}{obs}{reward}{training}", verbose=2)
    # rl.train()

    # Delete the reference to the ReinforcementLearning object so it can be garbage collected
    # before another ReinforcementLearning object is created in the next iteration
    # This avoids having two replay buffers occupying memory at the same time
    # del rl

# In total 2 * 3 - 1 = 5 agents ==> 5 * 5 = 25 hours of training

experiment = {
    "description": """RL Experiment 4
This script trains different versions of the agent having 99 actions per block""",
    "agents": agents
}
with open('experiments/experiment4.json', 'w') as f:
    json.dump(experiment, f, indent=4)
