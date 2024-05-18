"""
rl_baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baselines
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


if __name__ == "__main__":

    # barchart_instances(['MILP', 'PSO'])
    plot_history_values_instances(['MILP', 'PSO'], save_fig=True)
