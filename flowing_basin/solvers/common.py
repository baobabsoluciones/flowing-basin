import os
from flowing_basin.core import Solution
from matplotlib import pyplot as plt
import numpy as np
import re


BASELINES_FOLDER = os.path.join(os.path.dirname(__file__), "../rl_data/baselines")


def get_all_baselines(general_config: str) -> list[Solution]:

    """
    Get all baseline solutions by scanning the baselines folder.

    :param general_config: General configuration (e.g. "G1")
    """

    parent_dir = os.path.join(BASELINES_FOLDER, general_config)
    return scan_baselines(parent_dir)


def get_all_baselines_folder(folder_name: str, general_config: str) -> list[Solution]:

    """
    Get all baseline solutions in the folder `folder_name`.

    :param folder_name: Name of the folder in which to find the baselines
    :param general_config: General configuration (e.g. "G1")
    """

    parent_dir = os.path.join(BASELINES_FOLDER, folder_name, general_config)
    return scan_baselines(parent_dir)


def scan_baselines(folder_path: str) -> list[Solution]:

    """
    Scan the baselines in the given folder.

    :param folder_path: Path to the folder in which to scan the baselines
    """

    sols = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            full_path = os.path.join(folder_path, file)
            sol = Solution.from_json(full_path)
            sols.append(sol)
    return sols


def barchart_instances(values: dict[str, dict[str, float]], value_type: str, title: str, general_config: str):

    """
    Plot a barchart with the value of each solver at every instance.
    The values may be incomes, rewards, or anything else.
    The 'solvers' may actually be something else, e.g. different reward configurations.

    :param values: dict[solver, dict[instance, value]]
    :param value_type: Indicate which value is being plotted (income, reward...)
    :param title: String that will appear on the title
    :param general_config: General configuration (e.g. "G1")
    """

    solvers = list(values.keys())
    bar_width = 0.4 * 2. / len(solvers)
    offsets = [i * bar_width for i in range(len(solvers))]

    # Instances are ordered according to their instance percentile number (e.g. 'Percentile70' -> 70)
    instances = list(values[solvers[0]].keys())
    instances.sort(key=lambda instance_name: int(re.search(r'\d+', instance_name).group()))
    x_values = np.arange(len(instances))

    # Plot the bars for all instances, one solver at a time
    fig, ax = plt.subplots()
    for solver, offset in zip(solvers, offsets):
        # Items must be ordered according to their instance percentile number (e.g. 'Percentile70' -> 70)
        # in order to match the x labels
        sorted_values = dict(sorted(
            values[solver].items(), key=lambda item: int(re.search(r'\d+', item[0]).group())
        ))  # noqa
        print(f"Histogram values for {solver}:", x_values + offset, list(sorted_values.values()))
        ax.bar(x_values + offset, list(sorted_values.values()), width=bar_width, label=solver)
    ax.set_xticks(x_values + bar_width / 2)
    ax.set_xticklabels(instances, rotation='vertical')

    ax.set_xlabel('Instances')
    ax.set_ylabel(value_type)
    ax.set_title(f'Bar chart of {title} for all instances in {general_config}')
    ax.legend()

    plt.tight_layout()
    plt.show()
