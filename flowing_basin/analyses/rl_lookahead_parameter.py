from flowing_basin.core import Instance
from cornflow_client.core.tools import load_json
from math import ceil
import pandas as pd

path_constants = "../data/rl_training_data/constants.json"
constants = load_json(path_constants)
inst_const = Instance.from_dict(constants)

path_training_data = "../data/rl_training_data/training_data.pickle"
training_data = pd.read_pickle(path_training_data)

# Average incoming flows

avg_in_flow = dict()
avg_in_flow["dam1"] = training_data["incoming_flow"].mean() + training_data["dam1_unreg_flow"].mean()
avg_in_flow["dam2"] = training_data["dam1_turbined_flow"].mean() + training_data["dam2_unreg_flow"].mean()
print(avg_in_flow)

# Average time required to fill or empty dam

avg_time_fill = dict()
avg_time_empty = dict()

for dam_id in inst_const.get_ids_of_dams():

    vol_change = inst_const.get_max_vol_of_dam(dam_id) - inst_const.get_min_vol_of_dam(dam_id)
    avg_time_fill[dam_id] = vol_change / avg_in_flow[dam_id]

    max_flow = inst_const.get_max_flow_of_channel(dam_id)
    avg_time_empty[dam_id] = vol_change / (max_flow - avg_in_flow[dam_id])

print(avg_time_fill)
print(avg_time_empty)

# Required 'look ahead' parameter

look_ahead = max(max(avg_time_fill[dam_id], avg_time_empty[dam_id]) for dam_id in inst_const.get_ids_of_dams())
look_ahead = look_ahead / inst_const.get_time_step_seconds()
look_ahead = ceil(look_ahead)
print(look_ahead)
