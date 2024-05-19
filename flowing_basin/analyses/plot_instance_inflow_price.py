"""
plot_instance_inflow_price.py
This script plots the inflow and price level of every instance Percentile00, ..., Percentile100
"""

import matplotlib.pyplot as plt
import numpy as np
from flowing_basin.solvers.common import get_all_instances


if __name__ == "__main__":

    fig, ax = plt.subplots()
    twin_ax = ax.twinx()

    instances = get_all_instances(num_dams=2)
    instance_names = [instance.get_instance_name() for instance in instances]
    instance_inflows = [instance.get_total_avg_inflow() for instance in instances]
    instance_prices = [instance.get_avg_price() for instance in instances]

    bar_width = 0.4
    x_values = np.arange(len(instances))

    ax.bar(x_values, instance_inflows, width=bar_width, label="Inflow", color="blue")
    twin_ax.bar(x_values + bar_width, instance_prices, width=bar_width, label="Price", color="red")

    ax.set_xticks(x_values + bar_width / 2)
    ax.set_xticklabels(instance_names, rotation='vertical')

    ax.set_xlabel('Instances')
    ax.set_ylabel('Average inflow (m3/s)')
    twin_ax.set_ylabel('Average price (€/MWh)')

    ax.set_title(f'Bar chart of inflow and price for all instances')
    ax.legend()
    twin_ax.legend()

    plt.tight_layout()
    plt.savefig(f"instance_charts/instance_inflow_price.png")
    plt.show()
