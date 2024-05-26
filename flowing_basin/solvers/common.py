import os
from typing import Any
from flowing_basin.core import Instance, Solution
from flowing_basin.tools import PowerGroup
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import csv
from datetime import timedelta


CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), "../data/constants/constants_{num_dams}dams.json")
BASELINES_FOLDER = os.path.join(os.path.dirname(__file__), "../rl_data/baselines")
GENERAL_CONFIGS = ['G0', 'G1', 'G2', 'G3']


def get_episode_length(constants: Instance, num_days: int = 1) -> int:
    """
    Get the length of an episode (number of periods in a day + impact buffer).
    """
    impact_buffer = max(constants.get_relevant_lags_of_dam(dam_id)[0] for dam_id in constants.get_ids_of_dams())
    day_periods = timedelta(days=num_days) // timedelta(seconds=constants.get_time_step_seconds())
    length_episode = day_periods + impact_buffer
    return length_episode


def get_real_max_flow(constants: Instance, dam_id: str) -> float:
    """Get the real max flow of the given dam, obtained by interpolating the max volume."""
    real_max_flow = constants.get_max_flow_of_channel(dam_id)
    points = constants.get_flow_limit_obs_for_channel(dam_id)
    if points is not None:  # Get the real max_flow (dam2)
        max_vol = constants.get_max_vol_of_dam(dam_id)
        real_max_flow = min(real_max_flow, np.interp(max_vol, points["observed_vols"], points["observed_flows"]))
    return real_max_flow


def get_turbine_count_intervals(constants: Instance, first_flow_padding: float = 0.005):
    """
    For every dam, get the turbine count, first flow, and last flow of each interval in the power group curve
    :param constants: Instance object with at least the constant values
    :param first_flow_padding: Parameter to keep all the flows in the interval within the same number of turbines
    :return: {dam_id: [(turbine_count, first_flow, last_flow),]}
    """
    turbine_count_flows = dict()
    for dam_id in constants.get_ids_of_dams():

        max_flow = get_real_max_flow(constants=constants, dam_id=dam_id)
        startup_flows = constants.get_startup_flows_of_power_group(dam_id)
        shutdown_flows = constants.get_shutdown_flows_of_power_group(dam_id)
        turbined_bin_flows, turbined_bin_groups = PowerGroup.get_turbined_bins_and_groups(
            startup_flows, shutdown_flows, epsilon=0.  # epsilon=0. to avoid touching the limit zones
        )

        i = 0
        turbined_bin_flows = [0.] + turbined_bin_flows.tolist() + [max_flow]  # noqa
        turbine_count_flows[dam_id] = []
        while i < len(turbined_bin_flows) - 1:
            first_flow = turbined_bin_flows[i]
            if first_flow != 0.:
                first_flow += first_flow_padding
            last_flow = turbined_bin_flows[i + 1]
            turbine_count_flows[dam_id].append((turbined_bin_groups[i], first_flow, last_flow))
            i = i + 1

    return turbine_count_flows


def get_all_instances(num_dams: int) -> list[Instance]:
    """
    Get instances Percentile00, Percentile10, ..., Percentile100, from driest to rainiest.
    """
    instances = [
        Instance.from_name(f"Percentile{percentile:02}", num_dams=num_dams) for percentile in range(0, 110, 10)
    ]
    return instances


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


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the confidence interval for a list of values.

    :param data: Array of shape (num_cases, num_replications).
    :param confidence: The confidence level for the interval.
    :return:
        Tuple with two arrays of shape (num_cases,).
        These arrays represent the lower and upper bounds, respectively, of the confidence interval for each case.
    """

    if data.ndim != 2:
        raise ValueError(
            f"Input data must be a 2D array with (num_cases, num_replications), but the given shape is {data.shape}"
        )

    num_replications = data.shape[1]
    means = np.mean(data, axis=1)

    # Calculate the standard error of the mean, s / √n
    sems = stats.sem(data, axis=1)

    # Get the quantile corresponding to the given confidence, using the quantile function (qt) of the t_n-1 distribution
    alpha = 1 - confidence
    quantile = stats.t.ppf(1 - alpha / 2, num_replications - 1)

    margin_of_error = sems * quantile
    lower_bound = means - margin_of_error
    upper_bound = means + margin_of_error

    return lower_bound, upper_bound


def lighten_color(color, amount=0.5):
    """
    Function from StackOverflow (https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib)
    Returns a lighter (amount<1) or darker (amount>1) version of the color
    Examples:
    >> lighten_color('green', 0.3)
    # Returns a color that is like the 'green' color, but lighter
    >> lighten_color('green', 1.3)
    # Returns a color that is like the 'green' color, but darker
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def print_save_csv(rows: list[list[Any]], csv_filepath: str = None):
    """
    Save and print the given list of lists as a CSV file
    @param rows:
    @param csv_filepath:
    @return:
    """

    # Save results in .csv file
    if csv_filepath is not None:
        with open(csv_filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)

    # Print results
    for row in rows:
        row = [f'{el:.2f}' if isinstance(el, float) else el for el in row]
        print(','.join(row))


def extract_percentile(instance_name: str) -> int:
    """Extract the percentile number from the instance name: 'Percentile70' -> 70"""
    return int(instance_name.split('Percentile')[1])


def preprocess_values(values: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Sorts the dictionary by instance percentile number and returns the implicit solvers and instances.
    It also checks that all solvers have been used for the same instances.
    :param values: dict[solver, dict[instance, Any]]
    :return: List of solvers and list of instances
    """

    solvers = list(values.keys())
    for solver in solvers:
        values[solver] = dict(sorted(values[solver].items(), key=lambda x: extract_percentile(x[0])))  # noqa
    instances = list(values[solvers[0]].keys())

    if any(instances != list(values[solver].keys()) for solver in solvers[1:]):
        values_keys = {solver: list(values[solver].keys()) for solver in solvers}
        raise ValueError(f"The instances solved by every solver do not match: {values_keys}")

    return solvers, instances


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

    solvers, instances = preprocess_values(values)
    bar_width = 0.4 * 2. / len(solvers)
    offsets = [i * bar_width for i in range(len(solvers))]
    x_values = np.arange(len(instances))

    # Plot the bars for all instances, one solver at a time
    for solver, offset in zip(solvers, offsets):

        # Get the mean and, if they exist, upper and lower bounds
        values_mean = []
        values_lower = []
        values_upper = []
        for instance_name, instance_values in values[solver].items():
            if isinstance(instance_values, list):
                if len(instance_values) > 1:
                    values_mean.append(np.mean(instance_values))
                    lower, upper = confidence_interval(np.array(instance_values).reshape(1, -1))
                    values_lower.append(lower.item())
                    values_upper.append(upper.item())
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


