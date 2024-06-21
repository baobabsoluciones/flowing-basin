"""
RL Run Times

This script allows measuring the time needed to run an agent.
"""

# Disable `tensorboard` when we do not intend to save the agent,
# because Tensorboard always raises an error on debug mode
SAVE_AGENT = False
if not SAVE_AGENT:
    import sys
    sys.modules['torch.utils.tensorboard'] = None
from flowing_basin.solvers.rl import ReinforcementLearning
import time
import csv


if __name__ == "__main__":

    data = []
    agents = [
        "rl-A31G0O231R1T1002",
        "rl-A113G1O2R22T302",
        "rl-A21G2O3R1T748",
        "rl-A21G3O3R1T74"
    ]
    for agent in agents:
        rl = ReinforcementLearning(agent, verbose=0)
        start = time.perf_counter()
        rl.run_agent([f"Percentile{i*10:02}" for i in range(0, 11)])
        elapsed_time = (time.perf_counter() - start) / 11
        print(agent, "Time:", elapsed_time)
        data.append([agent, elapsed_time])

    for general in ['G0', 'G01', 'G1', 'G2', 'G21', 'G3']:
        rl = ReinforcementLearning(f"rl-A1{general}O2R1T1", verbose=0)
        start = time.perf_counter()
        for instance in rl.get_all_fixed_instances(rl.config.num_dams):
            rl.run_named_policy("greedy", instance)
        elapsed_time = (time.perf_counter() - start) / 11
        print(f"Greedy_{general}", "Time:", elapsed_time)
        data.append([f"Greedy_{general}", elapsed_time])

    csv_file = 'reports/rl_agent_times.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Agent", "Average time (s)"])
        writer.writerows(data)
