"""
baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baseline, Baselines
from flowing_basin.solvers.common import print_save_csv
import matplotlib.pyplot as plt


GENERAL_CONFIGS = ['G0', 'G1', 'G2', 'G3']


def barchart_instances(
        solvers: list[str], include_folders: list[str] = None, general_configs: list[str] = None, save_fig: bool = False
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS

    num_configs = len(general_configs)
    layout = [(1, 1), (1, 2), (2, 2), (2, 2)][num_configs - 1]
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(6 * layout[1], 6 * layout[0]))

    axes = axes.flatten()
    for i, ax in enumerate(axes[:num_configs]):
        Baselines(solvers=solvers, include_folders=include_folders, general_config=f'G{i}').barchart_instances_ax(ax)
    plt.tight_layout()

    solvers_title = "_".join(solvers)
    configs_title = ("_" + "_".join(general_configs)) if general_configs != GENERAL_CONFIGS else ""
    if save_fig:
        plt.savefig(f"reports/barchart_instances_{solvers_title}{configs_title}.png")
    plt.show()


def plot_history_values_instances(solvers: list[str], save_fig: bool = False):

    solvers_title = "_".join(solvers)
    for general_config in GENERAL_CONFIGS:
        filename = f"reports/history_curves_{solvers_title}_{general_config}.png" if save_fig else None
        Baselines(solvers=solvers, general_config=general_config).plot_history_values_instances(filename=filename)


def csv_instance_final_values(
        solvers: list[str], reference: str = None, general_configs: list[str] = None, save_csv: bool = False, **kwargs
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS

    rows_total = []
    for i, general_config in enumerate(general_configs):
        baselines = Baselines(solvers=solvers, general_config=general_config, **kwargs)
        rows = baselines.get_csv_instance_final_values(reference)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'

    solvers_title = "_".join(solvers)
    configs_title = ("_" + "_".join(general_configs)) if general_configs != GENERAL_CONFIGS else ""
    reference_title = f"_ref_{reference}" if reference is not None else ""
    csv_filename = f"reports/final_values_{solvers_title}{configs_title}{reference_title}.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


def csv_instance_final_timestamps(
        solvers: list[str], include_folders: list[str] = None, general_configs: list[str] = None,
        save_csv: bool = False
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS

    rows_total = []
    for i, general_config in enumerate(general_configs):
        baselines = Baselines(solvers=solvers, include_folders=include_folders, general_config=general_config)
        rows = baselines.get_csv_instance_final_timestamps()
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'

    solvers_title = "_".join(solvers)
    configs_title = ("_" + "_".join(general_configs)) if general_configs != GENERAL_CONFIGS else ""
    csv_filename = f"reports/final_timestamps_{solvers_title}{configs_title}.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


def csv_instance_smoothing_violations(solvers: list[str], in_percentage: bool = True, save_csv: bool = False):

    rows_total = []
    for i, general_config in enumerate(GENERAL_CONFIGS):
        rows = Baselines(
            solvers=solvers, general_config=general_config
        ).get_csv_instance_smoothing_violations(in_percentage)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'
    solvers_title = "_".join(solvers)
    pct_title = "_pct" if in_percentage else ""
    csv_filename = f"reports/smoothing_violations_{solvers_title}{pct_title}.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


def csv_final_milp_gap(save_csv: bool = False):

    rows_total = []
    for i, general_config in enumerate(GENERAL_CONFIGS):
        rows = Baselines(solvers=['MILP'], general_config=general_config).get_csv_milp_final_gaps()
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'
    csv_filename = f"reports/final_milp_gap.csv" if save_csv else None
    print_save_csv(rows_total, csv_filepath=csv_filename)


if __name__ == "__main__":

    # barchart_instances(['PSO'], include_folders=['tuned'], general_configs=['G0', 'G1'], save_fig=True)
    # csv_instance_final_values(['PSO'], include_folders=['tuned'], reference='PSO', general_configs=['G0', 'G1'], save_csv=True)
    # csv_instance_final_timestamps(['MILP', 'PSO'], save_csv=True)
    # plot_history_values_instances(['MILP', 'PSO', 'rl-greedy'], save_fig=True)
    # csv_instance_final_values(['MILP', 'PSO', 'rl-greedy'], save_csv=True)
    # csv_instance_final_values(['MILP', 'PSO', 'rl-greedy'], reference='rl-greedy', save_csv=True)
    # csv_instance_smoothing_violations(['MILP', 'PSO', 'rl-greedy'], in_percentage=False, save_csv=True)
    # csv_instance_smoothing_violations(['MILP', 'PSO', 'rl-greedy'], in_percentage=True, save_csv=True)
    # csv_final_milp_gap(save_csv=True)

    # Compare rl-greedy with stored Heuristic solutions
    csv_instance_final_values(['rl-greedy', 'Heuristic'], general_configs=['G9'])
    csv_instance_final_values(['rl-greedy', 'Heuristic'], reference='rl-greedy', general_configs=['G9'])

    # Compare rl-greedy with fresh new Heuristic solutions
    # sols = Baseline(solver="Heuristic", general_config='G1').solve(save_sol=False)
    # csv_instance_final_values(['rl-greedy'], reference='rl-greedy', general_configs=['G1'], include_solutions=sols)

    # Run multiple replications of RBO
    # sols = Baseline(solver="RBO", general_config='G1').solve(save_sol=False, num_replications=2)
    # csv_instance_final_values([], general_configs=['G1'], include_solutions=sols)
