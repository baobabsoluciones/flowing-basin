from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

flows = np.array(
    [
        [3., 4., 5.],
        [5., 6., 7.],
        [7., 8., 9.],
        [3., 4., 5.],
        [10., 12., 12.],
    ]
)

num_scenarios = flows.shape[-1]
dam = instance.get_ids_of_dams()[0]
print(dam)
initial_lags = instance.get_initial_lags_of_channel(dam)
past_flows = np.repeat([initial_lags], repeats=num_scenarios, axis=0)
print(f"{past_flows=}")

power_group = PowerGroup(
    idx=dam,
    past_flows=past_flows,
    instance=instance,
    paths_power_models=paths_power_models,
    num_scenarios=num_scenarios,
)
print(f"{power_group.power=}")
print(f"{power_group.turbined_flow=}")
print(f"{power_group.previous_num_active_groups=}")
print(f"{power_group.num_active_groups=}")
print(f"{power_group.acc_num_startups=}")
print(f"{power_group.acc_num_times_in_limit=}")
print()

for flow in flows:

    past_flows = np.insert(past_flows, obj=0, values=flow, axis=1)
    past_flows = np.delete(past_flows, obj=past_flows.shape[1] - 1, axis=1)
    print(f"{past_flows=}")

    power_group.update(past_flows)
    print(f"{power_group.power=}")
    print(f"{power_group.turbined_flow=}")
    print(f"{power_group.previous_num_active_groups=}")
    print(f"{power_group.num_active_groups=}")
    print(f"{power_group.acc_num_startups=}")
    print(f"{power_group.acc_num_times_in_limit=}")
    print()
