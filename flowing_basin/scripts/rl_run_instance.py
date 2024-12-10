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
        "rl-A1G0O2R1T1-2",
        "rl-A31G0O231R1T1002-1",
        "rl-A21G0O3R1T3-2",
        "rl-A1G2O2R1T14-2",
        "rl-A31G2O231R1T1402-2",
        "rl-A21G2O3R1T74-1"
    ]

    # Greedy performance
    greedy_income = dict()
    for general in ['G0', 'G2']:
        rl = ReinforcementLearning(f"rl-A1{general}O2R1T1", verbose=0)
        start = time.perf_counter()
        for instance in rl.get_all_fixed_instances(rl.config.num_dams):
            instance_name = instance.get_instance_name()
            run = rl.run_named_policy("greedy", instance)
            obj_fun = run.solution.get_objective_function()
            greedy_income[general, instance_name] = obj_fun
            data.append([f"Greedy_{general}", instance_name, obj_fun, 0.])

    # MILP performance
    for general in ['G0', 'G2']:
        sols = ReinforcementLearning.get_all_baselines(general)
        for sol in sols:
            if sol.get_solver() == "MILP":
                instance_name = sol.get_instance_name()
                obj_fun = sol.get_objective_function()
                obj_fun_greedy = greedy_income[general, instance_name]
                pct_over_greedy = (obj_fun - obj_fun_greedy) / obj_fun_greedy
                data.append([f"MILP_{general}", instance_name, obj_fun, pct_over_greedy])

    # Agent performance
    for agent in agents:
        rl = ReinforcementLearning(agent, verbose=0)
        general = rl.config_names['G']
        runs = rl.run_agent([f"Percentile{i*10:02}" for i in range(0, 11)])
        for run in runs:
            instance_name = run.instance.get_instance_name()
            obj_fun = run.solution.get_objective_function()
            obj_fun_greedy = greedy_income[general, instance_name]
            pct_over_greedy = (obj_fun - obj_fun_greedy) / obj_fun_greedy
            data.append([agent, instance_name, obj_fun, pct_over_greedy])

    csv_file = 'reports/rl_run_instance_MDPI_with_MILP.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Agent", "Instance", "Income", "% over greedy"])
        writer.writerows(data)
