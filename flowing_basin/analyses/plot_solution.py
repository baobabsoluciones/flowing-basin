"""
plot_solution.py
"""

from flowing_basin.solvers.common import get_baseline
import matplotlib.pyplot as plt

INSTANCE = "Percentile70"
SOLVER = "MILP"
CONFIG = "G0"
FILENAME = f"plot_solution/solution_{CONFIG}_{INSTANCE}_{SOLVER}"
DAM_NAMES = {'dam1': 'the first reservoir', 'dam2': 'the second reservoir'}

if __name__ == "__main__":

    solution = get_baseline(general_config=CONFIG, instance=INSTANCE, solver=SOLVER)
    fig, axs = plt.subplots(1, solution.get_num_dams(), figsize=(6 * solution.get_num_dams(), 5))
    for i, dam_id in enumerate(solution.get_ids_of_dams()):
        solution.plot_solution_for_dam(dam_id=dam_id, ax=axs[i], dam_name=DAM_NAMES[dam_id])
    plt.savefig(FILENAME + '.eps')
    plt.savefig(FILENAME + '.png')
    plt.show()
