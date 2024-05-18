"""
plot_channel.py
Plot the flow limit of dam2 with respect to its volume
"""

from cornflow_client.core.tools import load_json
from flowing_basin.core import Instance
from flowing_basin.solvers.common import CONSTANTS_PATH
from matplotlib import pyplot as plt
import numpy as np

constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=2)))
points = constants.get_flow_limit_obs_for_channel("dam2")
observed_vols = points["observed_vols"]
observed_flows = points["observed_flows"]

plt.plot(observed_vols, observed_flows)
plt.xlabel("Observed volume")
plt.ylabel("Observed flow")

max_vol = constants.get_max_vol_of_dam("dam2")
interp_flow = np.interp(max_vol, observed_vols, observed_flows).item()
plt.scatter(max_vol, interp_flow, color='red', label=f'Max Vol Point')
plt.annotate(str(round(max_vol, 2)), xy=(max_vol, 0), xytext=(max_vol, -0.5), ha='center', color='red')
plt.annotate(str(round(interp_flow, 2)), xy=(0, interp_flow), xytext=(-0.5, interp_flow), ha='right', color='red')
plt.plot([max_vol, max_vol], [0, interp_flow], color='red', linestyle='--')
plt.plot([0, max_vol], [interp_flow, interp_flow], color='red', linestyle='--')

plt.legend()
plt.show()
