from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

flows2test = [1, 2, 3, 5, 6, 7, 8]
num_scenarios = len(flows2test)

dam = instance.get_ids_of_dams()[0]
print(dam)
initial_lags = instance.get_initial_lags_of_channel(dam)
past_flows = np.repeat([initial_lags], repeats=num_scenarios, axis=0)
print(past_flows)

power_group = PowerGroup(
    idx=dam,
    past_flows=past_flows,
    instance=instance,
    paths_power_models=paths_power_models,
    num_scenarios=num_scenarios,
)

print(instance.get_startup_flows_of_power_group(dam), instance.get_shutdown_flows_of_power_group(dam))
print(flows2test)
print(power_group.get_num_active_power_groups(np.array(flows2test)))
