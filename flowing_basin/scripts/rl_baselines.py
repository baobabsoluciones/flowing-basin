"""
rl_baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baselines
from flowing_basin.solvers.common import print_save_csv
import matplotlib.pyplot as plt


GENERAL_CONFIGS = ['G0', 'G1', 'G2', 'G3']


def barchart_instances(solvers: list[str], save_fig: bool = False):

    num_configs = len(GENERAL_CONFIGS)
    fig, axes = plt.subplots(num_configs // 2, num_configs // 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        Baselines(solvers=solvers, general_config=f'G{i}').barchart_instances_ax(ax)
    plt.tight_layout()
    solvers_title = "_".join(solvers)
    if save_fig:
        plt.savefig(f"reports/barchart_instances_{solvers_title}.png")
    plt.show()


def plot_history_values_instances(solvers: list[str], save_fig: bool = False):

    solvers_title = "_".join(solvers)
    for general_config in GENERAL_CONFIGS:
        filename = f"reports/history_curves_{solvers_title}_{general_config}.png" if save_fig else None
        Baselines(solvers=solvers, general_config=general_config).plot_history_values_instances(filename=filename)


def csv_instance_final_values(solvers: list[str], reference: str = None, save_csv: bool = False):

    rows_total = []
    for general_config in GENERAL_CONFIGS:
        rows = Baselines(solvers=solvers, general_config=general_config).get_csv_instance_final_values(reference)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows)
    solvers_title = "_".join(solvers)
    reference_title = f"_ref_{reference}" if reference is not None else ""
    csv_filename = f"reports/final_values_{solvers_title}{reference_title}.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


def csv_instance_smoothing_violations(solvers: list[str], in_percentage: bool = True, save_csv: bool = False):

    rows_total = []
    for general_config in GENERAL_CONFIGS:
        rows = Baselines(
            solvers=solvers, general_config=general_config
        ).get_csv_instance_smoothing_violations(in_percentage)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows)
    solvers_title = "_".join(solvers)
    pct_title = "_pct" if in_percentage else ""
    csv_filename = f"reports/smoothing_violations_{solvers_title}{pct_title}.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


if __name__ == "__main__":

    # barchart_instances(['MILP', 'PSO', 'rl-greedy'], save_fig=True)
    # plot_history_values_instances(['MILP', 'PSO', 'rl-greedy'], save_fig=True)
    csv_instance_final_values(['MILP', 'PSO', 'rl-greedy'], save_csv=True)
    # csv_instance_smoothing_violations(['MILP', 'PSO', 'rl-greedy'], in_percentage=False, save_csv=True)
