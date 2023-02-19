from flowing_basin.core import Instance
from flowing_basin.solvers import Environment

instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}

env = Environment(
    instance=instance,
    paths_power_models=paths_power_models,
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
)
print("low:", env._low.tensor)
print("high:", env._high.tensor)
print("initial observation:", env.get_observation(normalize=False))
print("initial observation (normalized):", env.get_observation())
