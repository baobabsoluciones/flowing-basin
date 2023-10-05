from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np

instance = Instance.from_json("../instances/instances_base/instance1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

flows = np.array(
    [
        [1.25, 1.25, 11.27, 11.27],
        [6.26, 6.26, 11.27, 11.27],
        [6.26, 6.26, 11.27, 10.02],
        [5.,   6.26, 8.77,  8.77],
        [3.,   4.,   5.,    8.],
        [3.,   4.,   5.,    8.],
        [3.,   4.,   5.,    8.],
    ]
)

num_scenarios = flows.shape[-1]
dam = instance.get_ids_of_dams()[1]
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
    mode="linear"
)
print(f"{power_group.power=}")
print(f"{power_group.turbined_flow=}")
print(f"{power_group.previous_num_active_groups=}")
print(f"{power_group.num_active_groups=}")
print(f"{power_group.acc_num_startups=}")
print(f"{power_group.acc_num_times_in_limit=}")
print(power_group.turbined_bins)
print(power_group.turbined_bin_groups)
print()

for flow in flows:

    past_flows = np.insert(past_flows, obj=0, values=flow, axis=1)
    past_flows = np.delete(past_flows, obj=past_flows.shape[1] - 1, axis=1)
    print(f"{past_flows=}")

    price = instance.get_price(power_group.time + 1)
    print(power_group.time + 1, price)

    power_group.update(price=price, past_flows=past_flows)
    print(f"\t{power_group.power=}")
    print(f"\t{power_group.turbined_flow=}")
    print(f"\t{power_group.previous_num_active_groups=}")
    print(f"\t{power_group.num_active_groups=}")
    print(f"\t{power_group.acc_num_startups=}")
    print(f"\t{power_group.acc_num_times_in_limit=}")
    print()
