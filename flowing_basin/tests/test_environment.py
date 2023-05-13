from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Environment, RLConfiguration
import numpy as np
from stable_baselines3.common.env_checker import check_env


# TEST ENVIRONMENTS ---- #

# ENVIRONMENT 1 WITH INSTANCE 1
instance1 = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
config = RLConfiguration(
    volume_objectives={
        "dam1": 59627.42324,
        "dam2": 31010.43613642857
    },
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
    length_episodes=24 * 4 + 3,
)
env1 = Environment(
    instance=instance1,
    config=config,
    paths_power_models=paths_power_models,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
)

# Private attributes of the environment
print("low:", env1.get_obs_lower_limits())
print("high:", env1.get_obs_upper_limits())

# Initial observation
print("---- initial observation ----")
print("initial observation (not normalized):", env1.get_observation(normalize=False))
print("initial observation:", env1.get_observation())

# Decisions
actions = np.array([
    [0.5, 0.5],
    [-0.25, -0.25],
    [-0.25, -0.25],
])
for i, action in enumerate(actions):
    print(f">>>> decision {i}")
    next_obs, reward, done, _ = env1.step(action, normalize_obs=False)
    print("reward (not normalized):", reward * env1.instance.get_largest_price())
    print("reward:", reward)
    print("observation (not normalized):", next_obs)
    print("done:", done)
print(">>>> history:")
print(env1.river_basin.history.to_string())

# ENVIRONMENT 2 (WITH INSTANCE 2)
instance2 = Instance.from_json("../data/input_example3.json")
env2 = Environment(
    instance=instance2,
    config=config,
    paths_power_models=paths_power_models,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
)

# ENVIRONMENT 2 | Initial observation
print("---- ENVIRONMENT 2 | initial observation ----")
print("initial observation (not normalized):", env2.get_observation(normalize=False))
print("initial observation:", env2.get_observation())

# ENVIRONMENT 2 | First decision
print("---- ENVIRONMENT 2 | decision 1 ----")
action = np.array([0.5, 0.5])
next_obs, _, _, _ = env2.step(action, normalize_obs=False)
print("observation (not normalized):", next_obs)

# RESET ENVIRONMENT 1 - ENVIRONMENT 1 WITH INSTANCE 2
print("---- ENVIRONMENT 1 | RESET WITH ENVIRONMENT 2'S INSTANCE ----")
env1.reset(instance2)

# RESET | Initial observation
print("---- ENVIRONMENT 1 AFTER RESET | initial observation ----")
print("initial observation (not normalized):", env1.get_observation(normalize=False))
print("initial observation:", env1.get_observation())

# RESET | First decision
print("---- ENVIRONMENT 1 AFTER RESET | decision 1 ----")
action = np.array([0.5, 0.5])
next_obs, _, _, _ = env1.step(action, normalize_obs=False)
print("observation (not normalized):", next_obs)

# TEST CREATE INSTANCE METHOD ---- #

print("---- ENVIRONMENT 1 | RESET WITH RANDOM INSTANCE ----")
env1.reset()
print(env1.instance.check())
print(env1.instance.data)
print(env1.get_observation())

# CHECK ENVS WITH SB3 ---- #

check_env(env1)
check_env(env2)
