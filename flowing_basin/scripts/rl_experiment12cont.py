"""
RL Experiment 12 (cont.)
This script trains the agents A21 using 6 dams with original and triple network sizes.
"""

from flowing_basin.solvers.rl import ReinforcementLearning
import json

best_agents = [
    "rl-A21G2O3R1T74",
    "rl-A21G2O3R1T748",
    "rl-A21G3O3R1T74",
    "rl-A21G3O3R1T748",
]

experiment = {
    "description": """RL Experiment 12 (cont.)
This script trains the agents A21 using 6 dams with original and triple network sizes.""",
    "agents": best_agents
}
with open('experiments/experiment12cont.json', 'w') as f:
    json.dump(experiment, f, indent=4)
