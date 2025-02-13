"""
plot_inflow_price_instance.py
Plot the price and inflows of a single instance.
"""

from flowing_basin.core import Instance
from matplotlib import pyplot as plt
import numpy as np


INSTANCE = "Percentile50"
NUM_DAMS = 2
RESERVOIR_NAMES = {'dam1': 'first reservoir', 'dam2': 'second reservoir'}
FILENAME = f"plot_inflow_price_instance/inflow_price_{INSTANCE}_step"

if __name__ == "__main__":

    instance = Instance.from_name(INSTANCE, num_dams=NUM_DAMS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    ax_prices = axs[0]
    prices = instance.get_all_prices()
    ax_prices.plot(prices, linewidth=2, color="red", label="Price of energy")
    ax_prices.legend()
    ax_prices.set_ylabel("Price (€/MWh)")
    ax_prices.set_xlabel("Day Period")
    ax_prices.grid('on')

    ax_inflows = axs[1]
    incoming_flows = np.array(instance.get_all_incoming_flows())
    for dam_id in instance.get_ids_of_dams():
        inflows = np.array(instance.get_all_unregulated_flows_of_dam(dam_id))
        if instance.get_order_of_dam(dam_id) == 1:
            inflows += incoming_flows
            linestyle = '-'
        else:
            linestyle = '--'
        label = f"Inflow of the {RESERVOIR_NAMES[dam_id]}"
        ax_inflows.step(
            x=range(len(inflows)), y=inflows, where='post', linewidth=2, color="blue", linestyle=linestyle, label=label
        )
    ax_inflows.legend()
    ax_inflows.set_ylabel("Inflow (m3/s)")
    ax_inflows.set_xlabel("Day Period")
    ax_inflows.grid('on')

    plt.savefig(FILENAME + '.eps')
    plt.savefig(FILENAME + '.png')
    plt.show()
