from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin

PATHS_POWER_MODELS = {
    "dam1": "C:\\Users\\rodri\\OneDrive - Universidad Politécnica de Madrid\\General\\ml_models\\model_E1.sav",
    "dam2": "C:\\Users\\rodri\\OneDrive - Universidad Politécnica de Madrid\\General\\ml_models\\model_E2.sav",
}

instance = Instance.from_json("../data/input.json")

river_basin = RiverBasin(instance=instance, paths_power_models=PATHS_POWER_MODELS)
print("initial state:", river_basin.get_state())

flows = {"dam1": 6.79, "dam2": 6.58}
river_basin.update(flows)
print("state after first decision:", river_basin.get_state())

flows = {"dam1": 7.49, "dam2": 6.73}
river_basin.update(flows)
print("state after second decision:", river_basin.get_state())
