"""
plot_channel.py
Plot the flow limit of dam2 with respect to its volume
"""

from cornflow_client.core.tools import load_json
from flowing_basin.core import Instance
from flowing_basin.solvers.common import CONSTANTS_PATH
from matplotlib import pyplot as plt
import numpy as np

PLOT_MAX_VOL = False
FILENAME = "plot_channel/fig_channels_superscript3"
DAM_NAMES = {'dam1': 'the first subsystem', 'dam2': 'the second subsystem'}

if __name__ == "__main__":

    constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=2)))

    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 5), sharey='all')
    for i, dam_id in enumerate(constants.get_ids_of_dams()):

        ax = axs[i]
        points = constants.get_flow_limit_obs_for_channel(dam_id)
        if points is not None:
            points_adjusted = [
                (vol, flow) for vol, flow in zip(points["observed_vols"], points["observed_flows"])
                if vol < constants.get_max_vol_of_dam(dam_id)
            ]
            observed_vols, observed_flows = zip(*points_adjusted)
            ax.plot(observed_vols, observed_flows, color='blue', linewidth=2)
            if PLOT_MAX_VOL:
                max_vol = constants.get_max_vol_of_dam("dam2")
                interp_flow = np.interp(max_vol, observed_vols, observed_flows).item()
                ax.scatter(max_vol, interp_flow, color='red', label=f'Max Vol Point')
                ax.annotate(str(round(max_vol, 2)), xy=(max_vol, 0), xytext=(max_vol, -0.5), ha='center', color='red')
                ax.annotate(str(round(interp_flow, 2)), xy=(0, interp_flow), xytext=(-0.5, interp_flow), ha='right',
                            color='red')
                ax.plot([max_vol, max_vol], [0, interp_flow], color='red', linestyle='--')
                ax.plot([0, max_vol], [interp_flow, interp_flow], color='red', linestyle='--')
        else:
            num = 50
            observed_vols = np.linspace(0, constants.get_max_vol_of_dam(dam_id), num)
            observed_flows = np.ones(num) * constants.get_max_flow_of_channel(dam_id)
            ax.plot(observed_vols, observed_flows, color='blue', linewidth=2)
        ax.set_title(f'Flow Limit function of {DAM_NAMES[dam_id]}')
        ax.set_xlabel("Volume (m$^3$)")
        if i == 0:
            ax.set_ylabel("Outflow limit (m$^3$/s)")

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(FILENAME + '.eps')
    plt.savefig(FILENAME + '.png')
    plt.show()
