"""
Script to train the agents for a RL experiment
"""

from flowing_basin.solvers.rl import ReinforcementLearning
import json

EXPERIMENT_CONFIG = "experiments/experiment6.json"

if __name__ == "__main__":

    with open(EXPERIMENT_CONFIG, 'r') as f:
        experiment_data = json.load(f)

    description = experiment_data["description"]
    agents = experiment_data["agents"]

    print("Description:")
    print("-" * 10)
    print(description)
    print("-" * 10)
    print("Agents:", agents)
    print("\nStarting training...\n")

    for agent in agents:

        rl = ReinforcementLearning(agent, verbose=3)

        # Some observation configurations require collecting before training
        if len(rl.config_names["O"]) >= 3:
            rl.collect_obs()

        rl.train()
        # When testing, run: rl.train(num_timesteps=15, save_agent=False)

        # Delete the reference to the ReinforcementLearning object so it can be garbage collected
        # before another ReinforcementLearning object is created in the next iteration
        # This avoids having two replay buffers occupying memory at the same time
        del rl
