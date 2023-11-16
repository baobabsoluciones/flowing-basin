from flowing_basin.core import Instance, Training
from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

PLOT_TRAINING_CURVE = False
SAVE_OBSERVATIONS = False
PATH_CONSTANTS = "../data/constants/constants_2dams.json"
PATH_TRAIN_DATA = "../data/history/historical_data_clean_train.pickle"
PATH_TEST_DATA = "../data/history/historical_data_clean_test.pickle"
PATH_OBSERVATIONS_SAVE = "reports/observations_data/observationsO2"
EVALUATION_INSTANCES = [
    f"../instances/instances_rl/instancePercentile{percentile:02}_expanded16steps_backforth.json"
    for percentile in range(0, 110, 10)
]
OPTIONS = dict(
    evaluation_instances=EVALUATION_INSTANCES,
    log_ep_freq=5,
    eval_ep_freq=5,
    eval_num_episodes=10,
    checkpoint_ep_freq=5,
)
FEATURE_EXTRACTOR = "CNN"
PROJECTOR_TYPE = "identity"

OBSERVATION_TYPE = "O2" if FEATURE_EXTRACTOR == "MLP" else "O1"
PATH_OBSERVATIONS = f"reports/observations_data/observations{OBSERVATION_TYPE}"
PATH_OBSERVATIONS_JSON = f"reports/observations_data/observations{OBSERVATION_TYPE}/config.json"

current_datetime = datetime.now().strftime('%Y-%m-%d %H.%M')
agent_folder = f"../solutions/rl_models/RL_model_{current_datetime}_f={FEATURE_EXTRACTOR}_p={PROJECTOR_TYPE}"

# Create configuration based on observations path
constants = Instance.from_dict(load_json(PATH_CONSTANTS))
config = RLConfiguration.from_json(PATH_OBSERVATIONS_JSON)
config.feature_extractor = FEATURE_EXTRACTOR
config.projector_type = PROJECTOR_TYPE
if config.projector_type != "identity":
    config.projector_bound = "max_min_per_component"
    config.projector_extrapolation = 0.5
    config.projector_explained_variance = .98
config.do_history_updates = False
config.update_observation_record = False
config.check()

train = RLTrain(
    config=config,
    path_constants=PATH_CONSTANTS,
    path_train_data=PATH_TRAIN_DATA,
    path_test_data=PATH_TEST_DATA,
    path_observations_folder=PATH_OBSERVATIONS,
    path_folder=agent_folder
)
train.solve(
    num_episodes=10,
    options=OPTIONS
)

# Save observation record for later PCA analysis
if SAVE_OBSERVATIONS:

    print("Observation record shape:", train.train_env.observation_record.shape)
    print("Observation record:", train.train_env.observation_record)

    os.makedirs(PATH_OBSERVATIONS_SAVE)
    np.save(os.path.join(PATH_OBSERVATIONS_SAVE, 'observations.npy'), train.train_env.observation_record)
    config.to_json(os.path.join(PATH_OBSERVATIONS_SAVE, 'config.json'))
    print(f"Created folder '{PATH_OBSERVATIONS_SAVE}'.")

if PLOT_TRAINING_CURVE:
    fig, ax = plt.subplots()
    training_data = Training.from_json(os.path.join(agent_folder, "training.json"))
    training_data.plot_training_curves(ax)
    plt.show()
