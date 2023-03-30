from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)

solution = Solution.from_json("../data/output_example1_PSO_k=1_m=0.2_i=100_p=20_c1=0.5_c2=0.3_w=0.9_v0.json")
print(river_basin.deep_update_flows(solution.to_nestedlist()))
