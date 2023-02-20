from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Environment
import torch


torch.set_printoptions(sci_mode=False)

# ENVIRONMENT 1 WITH INSTANCE 1
instance1 = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
env1 = Environment(
    instance=instance1,
    paths_power_models=paths_power_models,
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
)

# Private attributes of the environment
print("low:", env1.obs_low.tensor)
print("high:", env1.obs_high.tensor)

# Initial observation
print("---- initial observation ----")
obs = env1.get_observation()
print("initial observation:", obs.tensor)
print("initial observation (normalized):", obs.normalized)
print("initial reward:", env1.get_reward())

# First decision
print("---- decision 1 ----")
action = torch.tensor([6.79, 6.58])
reward, next_obs, done = env1.step(action)
print("reward:", reward)
print("observation:", next_obs.tensor)
print("done:", done)

# Second decision
print("---- decision 2 ----")
action = torch.tensor([7.49, 6.73])
reward, next_obs, done = env1.step(action)
print("reward:", reward)
print("observation:", next_obs.tensor)
print("done:", done)

# ENVIRONMENT 2 (WITH INSTANCE 2)
instance2 = Instance.from_json("../data/input_example3.json")
env2 = Environment(
    instance=instance2,
    paths_power_models=paths_power_models,
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
)

# ENVIRONMENT 2 | Initial observation
print("---- ENVIRONMENT 2 | initial observation ----")
obs = env2.get_observation()
print("initial observation:", obs.tensor)
print("initial observation (normalized):", obs.normalized)

# ENVIRONMENT 2 | First decision
print("---- ENVIRONMENT 2 | decision 1 ----")
action = torch.tensor([6.79, 6.58])
_, next_obs, _ = env2.step(action)
print("observation:", next_obs.tensor)

# RESET ENVIRONMENT 1 - ENVIRONMENT 1 WITH INSTANCE 2
print("---- RESET ----")
env1.reset(instance2)

# RESET | Initial observation
print("---- AFTER RESET | initial observation ----")
obs = env1.get_observation()
print("initial observation:", obs.tensor)
print("initial observation (normalized):", obs.normalized)

# RESET | First decision
print("---- AFTER RESET | decision 1 ----")
action = torch.tensor([6.79, 6.58])
_, next_obs, _ = env1.step(action)
print("observation:", next_obs.tensor)
