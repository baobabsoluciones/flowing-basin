"""
RL Greedy Percentages
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance
from matplotlib import pyplot as plt

instances = [Instance.from_name(f"Percentile{percentile:02}") for percentile in range(0, 110, 10)]
# instances = [Instance.from_name("Percentile50")]
general_configs = ["G0", "G1"]
percentages = [i / 10 for i in range(11)]
# percentages = [0.8]

for general in general_configs:

    # Greedy average income
    incomes = []
    rl = ReinforcementLearning(f"A1{general}O2R1T2", verbose=2)
    for pct in percentages:
        policy = f"greedy_{pct}"
        pct_incomes = []
        for instance in instances:
            sol = rl.run_named_policy(policy_name=policy, instance=instance)
            # for dam_id in sol.get_ids_of_dams():
            #     fig, ax = plt.subplots()
            #     sol.plot_solution_for_dam(dam_id, ax)
            #     plt.show()
            income = sol.get_objective_function()
            pct_incomes.append(income)
        avg_income = sum(pct_incomes) / len(pct_incomes)
        incomes.append(avg_income)

    fig, ax = plt.subplots()
    ax.set_xlabel("Percentage of flow")
    ax.set_ylabel("Average income (€)")
    ax.set_title(f"Income with fixed actions in environment {general}")
    ax.plot(percentages, incomes, label="rl-greedy_{pct}")
    plt.show()
