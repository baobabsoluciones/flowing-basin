from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

# Initial state
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)
state = river_basin.get_state()
print("initial state:", state)
old_state = state

# States after decisions
decisions = [{"dam1": 6.79, "dam2": 6.58}, {"dam1": 7.49, "dam2": 6.73}]
for i, flows in enumerate(decisions):

    river_basin.update(flows)
    state = river_basin.get_state()
    print(f"state after decision {i}:", state)
    print(f"income obtained after decision {i} (EUR): "
          f"price (EUR/MWh) * power (MW) * time_step (hours) = "
          f"{state['price']} * ({state['dam1']['power']} + {state['dam2']['power']}) * {instance.get_time_step()/3600} = "
          f"{state['price'] * (state['dam1']['power'] + state['dam2']['power']) * instance.get_time_step()/3600}")

    # Check volumes are consistent throughout states
    dam1_vol = float(np.clip(
        old_state["dam1"]["vol"]
        + (
            old_state["incoming_flow"]
            + old_state["dam1"]["unregulated_flow"]
            - state["dam1"]["lags"][0]
        )
        * instance.get_time_step(),
        instance.get_min_vol_of_dam("dam1"), instance.get_max_vol_of_dam("dam1")
    ))
    dam2_vol = float(np.clip(
        old_state["dam2"]["vol"]
        + (
            old_state["dam1"]["turbined_flow"]
            + old_state["dam2"]["unregulated_flow"]
            - state["dam2"]["lags"][0]
        )
        * instance.get_time_step(),
        instance.get_min_vol_of_dam("dam2"), instance.get_max_vol_of_dam("dam2")
    ))
    assert (
        round(state["dam1"]["vol"], 4) == round(dam1_vol, 4)
    ), f"Volume of dam1 should be {dam1_vol} but it is {state['dam1']['vol']}"
    assert (
        round(state["dam2"]["vol"], 4) == round(dam2_vol, 4)
    ), f"Volume of dam2 should be {dam2_vol} but it is {state['dam2']['vol']}"

    old_state = state
