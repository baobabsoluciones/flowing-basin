"""
baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baseline, Baselines
from flowing_basin.solvers.common import print_save_csv, preprocess_values
import matplotlib.pyplot as plt


# General configs we want to plot
GENERAL_CONFIGS = ['G0', 'G01', 'G1', 'G2', 'G21', 'G3']


def barchart_instances(
        solvers: list[str], general_configs: list[str] = None, save_fig: bool = False, **kwargs
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS

    num_configs = len(general_configs)
    layout = [(1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3)][num_configs - 1]
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(6 * layout[1], 6 * layout[0]))

    axes = axes.flatten()
    baseline_solvers = []
    for config, ax in zip(general_configs, axes[:num_configs]):
        baselines = Baselines(solvers=solvers, general_config=config, **kwargs)
        for solver in baselines.solvers:
            if solver not in baseline_solvers:
                baseline_solvers.append(solver)
        baselines.barchart_instances_ax(ax)
    plt.tight_layout()

    solvers_title = "_".join(baseline_solvers)
    configs_title = ("_" + "_".join(general_configs)) if general_configs != GENERAL_CONFIGS else ""
    if save_fig:
        plt.savefig(f"reports/barchart_instances_{solvers_title}{configs_title}.png")
        plt.savefig(f"reports/barchart_instances_{solvers_title}{configs_title}.eps")
    plt.show()


def plot_history_values_instances(
        solvers: list[str], general_configs: list[str] = None, save_fig: bool = False, transpose: bool = True, **kwargs
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS

    if transpose:
        solvers_processed = None
        num_instances = 11
        num_configs = len(general_configs)
        fig, axes = plt.subplots(num_instances, num_configs, figsize=(3 * num_configs, 3 * num_instances))
        for j, general_config in enumerate(general_configs):
            baselines = Baselines(solvers=solvers, general_config=general_config, **kwargs)
            timestamps, values = baselines.get_solver_instance_history_values()
            solvers_processed, instances = preprocess_values(values)
            for i, instance_name in enumerate(instances):
                baselines.plot_history_values_instance_ax(
                    ax=axes[i, j], instance_name=instance_name, values=values, timestamps=timestamps,
                    solvers=solvers_processed, title=f"Instance {instance_name} in {general_config}"
                )
        plt.tight_layout()
        solvers_title = "_".join(solvers_processed)
        configs_title = ("_" + "_".join(general_configs)) if general_configs != GENERAL_CONFIGS else ""
        filename = f"reports/history_curves_{solvers_title}{configs_title}" if save_fig else None
        if filename is not None:
            plt.savefig(filename + ".png")
            plt.savefig(filename + ".eps")
        plt.show()

    else:
        solvers_title = "_".join(solvers)
        for general_config in general_configs:
            filename = f"reports/history_curves_{solvers_title}_{general_config}.png" if save_fig else None
            Baselines(solvers=solvers, general_config=general_config).plot_history_values_instances(filename=filename)


def csv_instance_final_values(
        solvers: list[str], reference: str = None, general_configs: list[str] = None,
        save_csv: bool = False, solvers_extra: list[str] = None, **kwargs
):

    if general_configs is None:
        general_configs = GENERAL_CONFIGS
    if solvers_extra is None:
        solvers_extra = []

    rows_total = []
    baseline_solvers = []
    for i, general_config in enumerate(general_configs):
        sols_extra = []
        for solver_extra in solvers_extra:
            sols_extra.extend(Baseline(solver=solver_extra, general_config=general_config).solve(save_sol=False))
        baselines = Baselines(solvers=solvers, general_config=general_config, include_solutions=sols_extra, **kwargs)
        for solver in baselines.solvers:
            if solver not in baseline_solvers:
                baseline_solvers.append(solver)
        rows = baselines.get_csv_instance_final_values(reference)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'

    solvers_title = "_".join(baseline_solvers)
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


def csv_instance_violations(
        solvers: list[str], concept: str, in_percentage: bool = True, save_csv: bool = False,
        num_decimals: int = 1, **kwargs
):

    rows_total = []
    for i, general_config in enumerate(GENERAL_CONFIGS):
        rows = Baselines(
            solvers=solvers, general_config=general_config, **kwargs
        ).get_csv_instance_violations(concept, in_percentage=in_percentage, num_decimals=num_decimals)
        for row in rows:
            row.insert(0, general_config)
        rows_total.extend(rows if i == 0 else rows[1:])  # Exclude header of subsequent rows
    rows_total[0][0] = 'Configuration'
    solvers_title = "_".join(solvers)
    pct_title = "_pct" if in_percentage else ""
    csv_filename = f"reports/violations_{concept}_{solvers_title}{pct_title}.csv" if save_csv else None
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

    # barchart_instances(['PSO', 'PSO-RBO'], solvers_best=['PSO', 'PSO-RBO'], save_fig=True)
    # barchart_instances(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], save_fig=True)
    # barchart_instances(['MILP', 'PSO', 'Heuristic', 'rl-greedy', 'rl-random'], include_folders=['tuned'], general_configs=['G0', 'G1', 'G2', 'G3'], save_fig=True)
    # csv_instance_final_values(['PSO-RBO'], include_folders=['tuned'], reference='PSO-RBO', save_csv=True)
    # csv_instance_final_values(['PSO', 'PSO-RBO'], solvers_best=['PSO', 'PSO-RBO'], reference='PSO (best)', save_csv=True)
    # csv_instance_final_values(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], save_csv=True)
    # csv_instance_final_values(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], reference='rl-greedy', save_csv=True)
    # csv_instance_final_timestamps(['MILP', 'PSO'], save_csv=True)
    plot_history_values_instances(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], save_fig=True)
    # csv_instance_violations(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], concept="max_relvar", in_percentage=True, save_csv=True)
    # csv_instance_violations(['MILP', 'PSO', 'PSO-RBO', 'rl-greedy'], solvers_best=['PSO', 'PSO-RBO'], concept="flow_smoothing", in_percentage=True, save_csv=True)
    # csv_final_milp_gap(save_csv=True)
