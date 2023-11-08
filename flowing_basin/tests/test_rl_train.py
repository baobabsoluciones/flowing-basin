from flowing_basin.core import Instance, Training
from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


PATH_CONSTANTS = "../data/constants/constants_2dams.json"
PATH_TRAIN_DATA = "../data/history/historical_data_clean_train.pickle"
PATH_TEST_DATA = "../data/history/historical_data_clean_test.pickle"

current_datetime = datetime.now().strftime('%Y-%m-%d %H.%M')
agent_folder = f"../solutions/rl_models/RL_model_{current_datetime}"

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
PLOT_TRAINING_CURVE = True

SAVE_OBSERVATIONS = False
PATH_OBSERVATIONS = "../analyses/rl_pca/observations_data/observationsO2.npy"
PATH_OBSERVATIONS_CONFIG = "../analyses/rl_pca/observations_data/observationsO2_config.json"

constants = Instance.from_dict(load_json(PATH_CONSTANTS))
config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    flow_smoothing=2,
    flow_smoothing_penalty=25,
    flow_smoothing_clip=False,
    action_type="exiting_flows",
    features=[
        "past_vols", "past_flows", "past_variations", "future_prices",
        "future_inflows", "past_turbined", "past_groups", "past_powers", "past_clipped",
    ],
    unique_features=["future_prices", ],
    num_steps_sight={
        ("past_flows", "dam1"): constants.get_verification_lags_of_dam("dam1")[-1] + 1,
        ("past_flows", "dam2"): constants.get_verification_lags_of_dam("dam2")[-1] + 1,
        "past_variations": 2, "future_prices": 16, "future_inflows": 16,
        "other": 1
    },
    projector='identity',
    length_episodes=24 * 4 + 3,
    do_history_updates=False,
    update_observation_record=SAVE_OBSERVATIONS,
)
train = RLTrain(
    config=config,
    path_constants=PATH_CONSTANTS,
    path_train_data=PATH_TRAIN_DATA,
    path_test_data=PATH_TEST_DATA,
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

    np.save(PATH_OBSERVATIONS, train.train_env.observation_record)
    print(f"Created .npy file '{PATH_OBSERVATIONS}'.")

    config.to_json(PATH_OBSERVATIONS_CONFIG)
    print(f"Created JSON file '{PATH_OBSERVATIONS_CONFIG}'.")

if PLOT_TRAINING_CURVE:
    fig, ax = plt.subplots()
    training_data = Training.from_json(os.path.join(agent_folder, "training.json"))
    training_data.plot_training_curves(ax)
    plt.show()
