"""
baseline_random_greedy.py
This script creates the baselines "rl-random" and "rl-greedy".
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.solvers.common import BASELINES_FOLDER, GENERAL_CONFIGS
from flowing_basin.core import Instance
import os.path

instance_names = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]
policies = ["random", "greedy"]
unbiased = True

for general in GENERAL_CONFIGS[2:]:
    baselines_folder = os.path.join(BASELINES_FOLDER, general)
    rl = ReinforcementLearning(f"A1{general}O2R1T2", verbose=2)
    for policy in policies:
        for instance_name in instance_names:
            print(f"{general} rl-{policy} Solving {instance_name}...")
            instance = Instance.from_name(instance_name, num_dams=rl.config.num_dams)
            sol = rl.run_named_policy(policy_name=policy, instance=instance, update_to_decisions=not unbiased)
            inconsistencies = sol.check()
            if inconsistencies:
                raise ValueError("There are inconsistencies in the solution:", inconsistencies)
            filename = os.path.join(
                baselines_folder, f"instance{instance_name}_RL{policy}{'_biased' if not unbiased else ''}.json"
            )
            sol.to_json(filename)
            print(f"{general} rl-{policy} Saved file {filename}.")
