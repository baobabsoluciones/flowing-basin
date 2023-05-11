from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Environment
import numpy as np
from stable_baselines3.common.env_checker import check_env


# TEST ENVIRONMENTS ---- #

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
    length_episodes=24 * 4 + 3,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
)

# Private attributes of the environment
print("low:", env1.get_obs_lower_limits())
print("high:", env1.get_obs_upper_limits())

# Initial observation
print("---- initial observation ----")
print("initial observation:", env1.get_observation(normalize=False))
print("initial observation (normalized):", env1.get_observation())

# First decision
decisions = [[6.79, 6.58], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73]]
for i, decision in enumerate(decisions):
    print(f">>>> decision {i}")
    action = np.array(decision)
    next_obs, reward, done, _ = env1.step(action, normalize_obs=False)
    print("reward:", reward)
    print("observation:", next_obs)
    print("done:", done)

# ENVIRONMENT 2 (WITH INSTANCE 2)
instance2 = Instance.from_json("../data/input_example3.json")
env2 = Environment(
    instance=instance2,
    paths_power_models=paths_power_models,
    num_prices=10,
    num_unreg_flows=10,
    num_incoming_flows=10,
    length_episodes=24 * 4 + 3,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
)

# ENVIRONMENT 2 | Initial observation
print("---- ENVIRONMENT 2 | initial observation ----")
print("initial observation:", env2.get_observation(normalize=False))
print("initial observation (normalized):", env2.get_observation())

# ENVIRONMENT 2 | First decision
print("---- ENVIRONMENT 2 | decision 1 ----")
action = np.array([6.79, 6.58])
next_obs, _, _, _ = env2.step(action, normalize_obs=False)
print("observation:", next_obs)

# RESET ENVIRONMENT 1 - ENVIRONMENT 1 WITH INSTANCE 2
print("---- RESET ----")
env1.reset(instance2)

# RESET | Initial observation
print("---- AFTER RESET | initial observation ----")
print("initial observation:", env1.get_observation(normalize=False))
print("initial observation (normalized):", env1.get_observation())

# RESET | First decision
print("---- AFTER RESET | decision 1 ----")
action = np.array([6.79, 6.58])
next_obs, _, _, _ = env1.step(action, normalize_obs=False)
print("observation:", next_obs)

# TEST CREATE INSTANCE METHOD ---- #

print("---- RESET WITH RANDOM INSTANCE ----")
env1.reset()
print(env1.instance.check())
print(env1.instance.data)
print(env1.get_observation())

# CHECK ENVS WITH SB3 ---- #

check_env(env1)
check_env(env2)
