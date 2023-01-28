from flowing_basin.core import Instance
from os import path

# '__file__' is the path of this file; 'path.dirname' returns the path to its directory
dir_path = path.dirname(path.dirname(__file__))
file_path = path.join(dir_path, "data/input.json")
print(file_path)
instance = Instance.from_json(file_path)

dam = 2
print("dictionary:", instance.data)
print("number of dams:", instance.get_num_dams())
print("initial volume:", instance.get_initial_vol_of_dam(dam))
print("min volume:", instance.get_min_vol_of_dam(dam))
print("max volume:", instance.get_max_vol_of_dam(dam))
print("unregulated flow:", instance.get_unregulated_flow_of_dam(dam))
print("initial lags:", instance.get_initial_lags_of_channel(dam))
print("relevant lags:", instance.get_relevant_lags_of_dam(dam))
print("points", instance.get_max_flow_points_of_channel(dam))
print("incoming flow:", instance.get_incoming_flow(0))
