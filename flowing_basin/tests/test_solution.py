from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin

instance = Instance.from_json("../instances/instances_base/instance1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models, mode = "linear")

solution = Solution.from_json("../solutions/instance1_LPmodel_2dams_1days_time2023-10-02_17-17.json")
print(solution.complies_with_flow_smoothing(2))
print(solution.check())
for dam_id in instance.get_ids_of_dams():
    print(solution.get_exiting_flows_of_dam(dam_id))

river_basin.deep_update_flows(solution.get_exiting_flows_array())
print(river_basin.history.to_string())
