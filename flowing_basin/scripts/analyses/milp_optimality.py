"""
This script evaluates the optimality of the "MILP" baseline.
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.solvers.common import GENERAL_CONFIGS
import matplotlib.pyplot as plt
import re


def extract_percentile(item):
    match = re.search(r'\d+', item[0])
    return int(match.group()) if match else float('inf')


for general in GENERAL_CONFIGS:

    instance_gap = []
    for baseline in ReinforcementLearning.get_all_baselines(general):
        if baseline.get_solver() == "MILP":
            instance_gap.append((baseline.get_instance_name(), baseline.get_final_gap_value()))
    instance_gap.sort(key=extract_percentile)  # Sort by instance percentile
    instances = [item[0] for item in instance_gap]
    final_gaps = [item[1] for item in instance_gap]
    print(f"Average final gap in {general}:", sum(final_gaps) / len(final_gaps))

    plt.bar(instances, final_gaps)
    plt.xticks(rotation='vertical')
    plt.xlabel('Instance')
    plt.ylabel('Final gap (%)')
    plt.title(f'Final gap of MILP solutions in {general}')
    plt.tight_layout()
    plt.show()
