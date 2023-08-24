from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)

solution = Solution.from_json("../data/output_instance1_LPmodel_V2_2dams_1days.json")
print(solution.check())
for dam_id in instance.get_ids_of_dams():
    print(solution.get_exiting_flows(dam_id))

# river_basin.deep_update_flows(solution.to_flows())
# print(river_basin.history)
