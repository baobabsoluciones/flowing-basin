from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from os import path

dir_path = path.dirname(path.dirname(__file__))
file_path = path.join(dir_path, "data/input.json")
instance = Instance.from_json(file_path)

river_basin = RiverBasin(instance=instance, path_power_model="")
print("initial state:", river_basin.get_state())

flows = {"dam1": 6.79, "dam2": 6.58}
river_basin.update(flows)
print("state after first decision:", river_basin.get_state())

flows = {"dam1": 7.49, "dam2": 6.73}
river_basin.update(flows)
print("state after second decision:", river_basin.get_state())
