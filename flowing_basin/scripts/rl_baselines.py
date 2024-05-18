"""
rl_baselines.py
Script to analyze existing solutions that act as baselines for RL
"""

from flowing_basin.solvers import Baselines
import matplotlib.pyplot as plt


def barchart_instances(solvers: list[str], save_fig: bool = False):

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        Baselines(solvers=solvers, general_config=f'G{i}').barchart_instances_ax(ax)
    plt.tight_layout()
    solvers_title = "_".join(solvers)
    if save_fig:
        plt.savefig(f"reports/barchart_instances_{solvers_title}.png")
    plt.show()


if __name__ == "__main__":

    barchart_instances(['MILP', 'PSO'])
