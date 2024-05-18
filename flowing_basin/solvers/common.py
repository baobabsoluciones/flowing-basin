import os
from flowing_basin.core import Solution
from matplotlib import pyplot as plt
import numpy as np
import re
import scipy.stats as stats


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


def confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate the confidence interval for a list of values.

    :param data: The list of values.
    :param confidence: The confidence level for the interval.
    :return: The lower and upper bounds of the confidence interval.
    """

    mean = np.mean(data)

    # Calculate the standard error of the mean, s / √n
    sem = stats.sem(data)

    # Get the quantile corresponding to the given confidence, using the quantile function (qt) of the t_n-1 distribution
    alpha = 1 - confidence
    quantile = stats.t.ppf(1 - alpha / 2, len(data) - 1)

    margin_of_error = sem * quantile
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def barchart_instances_ax(
        ax: plt.Axes, values: dict[str, dict[str, float | list[float]]],
        value_type: str, title: str, general_config: str
):

    """
    Plot a barchart in the gives Axes with the value of each solver at every instance.
    The values may be incomes, rewards, or anything else.
    The 'solvers' may actually be something else, e.g. different reward configurations.

    :param ax: matplotlib.pyplot Axes object
    :param values: dict[solver, dict[instance, value/s]]
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
    for solver, offset in zip(solvers, offsets):

        # Items must be ordered according to their instance percentile number (e.g. 'Percentile70' -> 70)
        # in order to match the x labels
        sorted_values = dict(sorted(
            values[solver].items(), key=lambda item: int(re.search(r'\d+', item[0]).group())
        ))  # noqa

        # Get the mean and, if they exist, upper and lower bounds
        values_mean = []
        values_lower = []
        values_upper = []
        for instance_name, instance_values in sorted_values.items():
            if isinstance(instance_values, list):
                if len(instance_values) > 1:
                    values_mean.append(np.mean(instance_values))
                    lower, upper = confidence_interval(instance_values)
                    values_lower.append(lower)
                    values_upper.append(upper)
                else:
                    values_mean.append(instance_values[0])
            elif isinstance(instance_values, float):
                values_mean.append(instance_values)
            else:
                raise ValueError(f"Invalid type {type(instance_values)} for {instance_values=}")
        print(f"Histogram values for {solver}: {values_mean=}, {values_lower=}, {values_upper=}")

        # Plot mean values as a barchart
        ax.bar(x_values + offset, values_mean, width=bar_width, label=solver)

        # Plot lower and upper bounds, if they exist
        if values_lower and values_upper:
            lower_errors = np.array(values_mean) - np.array(values_lower)
            upper_errors = np.array(values_upper) - np.array(values_mean)
            ax.errorbar(
                x_values + offset, values_mean, yerr=[lower_errors, upper_errors], fmt='none', ecolor='black', capsize=5
            )

    ax.set_xticks(x_values + bar_width / 2)
    ax.set_xticklabels(instances, rotation='vertical')

    ax.set_xlabel('Instances')
    ax.set_ylabel(value_type)
    ax.set_title(f'Bar chart of {title} for all instances in {general_config}')
    ax.legend()


def barchart_instances(**kwargs):

    """
    Plot a barchart with the value of each solver at every instance.
    The values may be incomes, rewards, or anything else.
    The 'solvers' may actually be something else, e.g. different reward configurations.

    :param kwargs: Parameters given to the `barchart_instances_ax` function.
    """

    _, ax = plt.subplots()
    barchart_instances_ax(ax, **kwargs)
    plt.tight_layout()
    plt.show()


