"""
river_basin_delays.py
This script analyzes how much time it takes for water to reach subsequent reservoirs
"""

from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np

TIMESTEPS = 99
DAMS = 6
TIMESTEP_START = 10
FLOW_TO_ASSIGN = 10

if __name__ == "__main__":

    # Create an instance with no incoming flows, except for the first dam at timestep TIMESTEP_START
    instance = Instance.from_name("Percentile50", num_dams=DAMS)
    instance.data['incoming_flows'] = [0 for _ in range(TIMESTEPS)]
    print(instance.data)
    for dam_id in instance.get_ids_of_dams():
        instance.data['dams'][dam_id]['flow_limit'] = {'exists': False}
        instance.data['dams'][dam_id]['unregulated_flows'] = [0 for _ in instance.get_all_unregulated_flows_of_dam(dam_id)]
        instance.data['dams'][dam_id]['initial_lags'] = [0 for _ in instance.get_initial_lags_of_channel(dam_id)]
        instance.data['dams'][dam_id]['initial_vol'] = instance.get_min_vol_of_dam(dam_id)
    instance.data['dams']["dam1"]['unregulated_flows'] = [
        0 if timestep != TIMESTEP_START else FLOW_TO_ASSIGN
        for timestep, unreg_flow in enumerate(instance.get_all_unregulated_flows_of_dam("dam1"))
    ]
    print(instance.data)

    # Assign FLOW_TO_ASSIGN to all dams
    river_basin = RiverBasin(instance=instance, mode="linear")
    decisions = np.zeros((TIMESTEPS, DAMS, 1))
    for dam_id in instance.get_ids_of_dams():
        decisions[:, instance.get_order_of_dam(dam_id) - 1, 0] = FLOW_TO_ASSIGN
    river_basin.deep_update_flows(decisions)
    print(river_basin.history.to_string())

    # Extract the exiting flows of each and save them as CSV
    for value in ["flow_clipped2", "turbined"]:
        history_subset = river_basin.history[[f"{dam_id}_{value}" for dam_id in instance.get_ids_of_dams()]]
        print(history_subset.to_string())
        history_subset.to_csv(f'river_basin_delays/river_basin_delays_{value}.csv')
