from flowing_basin.core import Instance
from flowing_basin.solvers import Environment
import torch


torch.set_printoptions(sci_mode=False)

# Create environment
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

# Private attributes of the environment
print("low:", env._low.tensor)
print("high:", env._high.tensor)

# Initial observation
obs = env.get_observation()
print("initial observation:", obs.tensor)
print("initial observation (normalized):", obs.normalized)
print("initial reward:", env.get_reward())

# First decision
print("---- decision 1 ----")
action = torch.tensor([6.79, 6.58])
reward, next_obs, done = env.step(action)
print("reward:", reward)
print("observation:", next_obs.tensor)
print("done:", done)

# Second decision
print("---- decision 2 ----")
action = torch.tensor([7.49, 6.73])
reward, next_obs, done = env.step(action)
print("reward:", reward)
print("observation:", next_obs.tensor)
print("done:", done)
