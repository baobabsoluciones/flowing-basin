from flowing_basin.core import Instance
from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector
import numpy as np
from stable_baselines3.common.env_checker import check_env
from datetime import datetime


INITIAL_ROW = "2021-03-27 11:30"
PATH_CONSTANTS = "../data/constants/constants_2dams.json"
PATH_HISTORICAL_DATA = "../data/history/historical_data_clean.pickle"
OBSERVATION_TYPE = "O1"
PATH_OBSERVATIONS = f"reports/observations_data/observations{OBSERVATION_TYPE}"
PATH_OBSERVATIONS_JSON = f"reports/observations_data/observations{OBSERVATION_TYPE}/config.json"

# ENVIRONMENT 1 (WITH INSTANCE 1)
constants = Instance.from_dict(load_json(PATH_CONSTANTS))
config = RLConfiguration.from_json(PATH_OBSERVATIONS_JSON)
config.feature_extractor = "CNN"
config.projector_type = "identity"
if config.projector_type != "identity":
    config.projector_bound = "max_min_per_component"
    config.projector_extrapolation = 0.5
    config.projector_explained_variance = .98
config.do_history_updates = True
config.update_observation_record = True
config.check()

projector = Projector.from_config(config, PATH_OBSERVATIONS)
env1 = RLEnvironment(
    config=config,
    projector=projector,
    path_constants=PATH_CONSTANTS,
    path_historical_data=PATH_HISTORICAL_DATA,
    initial_row=datetime.strptime(INITIAL_ROW, "%Y-%m-%d %H:%M"),
)

# Instance inside environment
# env1.instance.to_json("../instances/instances_rl/instance1_expanded16steps_backforth.json")
print("instance:", env1.instance.to_dict())
print("decision horizon:", env1.instance.get_decision_horizon())
print("impact horizon:", env1.instance.get_largest_impact_horizon())
print("information horizon:", env1.instance.get_information_horizon())

# ENVIRONMENT 1 OBSERVATION LIMITS
print("low:", env1.get_features_min_values())
print("high:", env1.get_features_max_values())

# ENVIRONMENT 1 INITIAL OBSERVATION
print("---- initial observation ----")
for dam_index, dam_id in enumerate(env1.instance.get_ids_of_dams()):
    env1.print_observation(dam_id)
    env1.print_observation(dam_id, normalize=True)

# ENVIRONMENT 1 | HARDCODED ACTIONS I
print("---- hardcoded actions I ----")
actions = np.array([
    [0.5, 0.5],
    [-0.25, -0.25],
    [-0.25, -0.25],
    [-1, -1],
])
for i, action in enumerate(actions):
    print(f">>>> decision {i}")
    next_obs, reward, done, _, _ = env1.step(action)
    print("reward details:", env1.get_reward_details())
    print("reward (not normalized):", reward * env1.instance.get_largest_price())
    print("reward:", reward)
    for dam_index, dam_id in enumerate(env1.instance.get_ids_of_dams()):
        env1.print_observation(dam_id)
        env1.print_observation(dam_id, normalize=True)
    print("done:", done)
print(">>>> history:")
print(env1.river_basin.history.to_string())
print(">>>> observation record:")
print(env1.observation_record)

# ENVIRONMENT 1 | HARDCODED ACTIONS II (FULL SOLUTION)
# print("---- hardcoded actions II (full solution) ----")
# env1.reset(initial_row=datetime.strptime(INITIAL_ROW, "%Y-%m-%d %H:%M"))
# decisionsVA = np.array(
#     [
#         [0.5, 0.5],
#         [-0.25, -0.25],
#         [-0.25, -0.25],
#     ]
# )
# padding = np.array(
#     [
#         [0, 0]
#         for _ in range(env1.instance.get_largest_impact_horizon() - decisionsVA.shape[0])
#     ]
# )
# decisionsVA = np.concatenate([decisionsVA, padding])
# for i, decision in enumerate(decisionsVA):
#     print(f">>>> decision {i}")
#     next_obs, reward, done, _, _ = env1.step(decision)
#     print("reward details:", env1.get_reward_details())
#     print("reward (not normalized):", reward * env1.instance.get_largest_price())
#     print("reward:", reward)
#     for dam_index, dam_id in enumerate(env1.instance.get_ids_of_dams()):
#         env1.print_observation(dam_id)
#         env1.print_observation(dam_id, normalize=True)
#     print("done:", done)
# print(">>>> history:")
# print(env1.river_basin.history.to_string())

# CHECK ENV1
# This must be done after the hardcoded actions because random actions are performed during the check
check_env(env1)

# ENVIRONMENT 2 (WITH INSTANCE 2)
# instance2 = Instance.from_json("../instances/instances_base/instance3.json")
# env2 = RLEnvironment(
#     instance=instance2,
#     config=config,
#     path_constants="../data/constants/constants_2dams.json",
#     path_historical_data="../data/history/historical_data.pickle",
# )
# check_env(env2)

# ENVIRONMENT 2 | Initial observation
# print("---- ENVIRONMENT 2 | initial observation ----")
# print("initial observation (not normalized):", env2.get_observation(normalize=False))
# print("initial observation:", env2.get_observation())

# ENVIRONMENT 2 | First decision
# print("---- ENVIRONMENT 2 | decision 1 ----")
# action = np.array([0.5, 0.5])
# next_obs, _, _, _ = env2.step(action, normalize_obs=False)
# print("observation (not normalized):", next_obs)

# RESET ENVIRONMENT 1 - ENVIRONMENT 1 (WITH INSTANCE 2)
# print("---- ENVIRONMENT 1 | RESET WITH ENVIRONMENT 2'S INSTANCE ----")
# env1.reset(instance2)

# RESET | Initial observation
# print("---- ENVIRONMENT 1 AFTER RESET | initial observation ----")
# print("initial observation (not normalized):", env1.get_observation(normalize=False))
# print("initial observation:", env1.get_observation())

# RESET | First decision
# print("---- ENVIRONMENT 1 AFTER RESET | decision 1 ----")
# action = np.array([0.5, 0.5])
# next_obs, _, _, _ = env1.step(action, normalize_obs=False)
# print("observation (not normalized):", next_obs)

# TEST 'CREATE INSTANCE' METHOD
# print("---- ENVIRONMENT 1 | RESET WITH RANDOM INSTANCE ----")
# env1.reset()
# print(env1.instance.check())
# print(env1.instance.data)
# print(env1.get_observation())
