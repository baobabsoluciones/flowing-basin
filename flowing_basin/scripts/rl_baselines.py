"""
RL Baselines
This script creates the baselines "rl-random" and "rl-greedy".
"""
import os.path

from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance

instances = [Instance.from_name(f"Percentile{percentile:02}") for percentile in range(0, 110, 10)]
general_configs = ["G0", "G1"]
policies = ["random", "greedy"]

for general in general_configs:
    baselines_folder = os.path.join(ReinforcementLearning.baselines_folder, general)
    rl = ReinforcementLearning(f"A1{general}O2R1T2", verbose=2)
    for policy in policies:
        for instance in instances:
            instance_name = instance.get_instance_name()
            print(f"{general} rl-{policy} Solving {instance_name}...")
            sol = rl.run_named_policy(policy_name=policy, instance=instance)
            filename = os.path.join(baselines_folder, f"instance{instance_name}_RL{policy}.json")
            sol.to_json(filename)
            print(f"{general} rl-{policy} Saved file {filename}.")
