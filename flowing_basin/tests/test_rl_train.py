from flowing_basin.core import Instance
from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime
from matplotlib import pyplot as plt

PATH_CONSTANTS = "../data/constants/constants_2dams.json"
PATH_TRAIN_DATA = "../data/history/historical_data_clean_train.pickle"
PATH_TEST_DATA = "../data/history/historical_data_clean_test.pickle"

current_datetime = datetime.now().strftime('%Y-%m-%d %H.%M')
filepath_agent = f"../solutions/rl_models/RL_model_{current_datetime}.zip"
filepath_config = f"../solutions/rl_models/RL_model_{current_datetime}_config.json"
filepath_training = f"../solutions/rl_models/RL_model_{current_datetime}_training.json"

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
        "past_vols", "past_flows", "past_variations", "past_prices", "future_prices", "past_inflows",
        "future_inflows", "past_turbined", "past_groups", "past_powers", "past_clipped", "past_periods"
    ],
    obs_box_shape=True,
    unique_features=[],
    num_steps_sight=16,
    length_episodes=24 * 4 + 3,
    log_ep_freq=5,
    eval_ep_freq=5,
    eval_num_episodes=10,
    do_history_updates=False,
)
train = RLTrain(
    config=config,
    path_constants=PATH_CONSTANTS,
    path_train_data=PATH_TRAIN_DATA,
    path_test_data=PATH_TEST_DATA
)

train.solve(
    num_episodes=200,
    path_agent=filepath_agent,
    periodic_evaluation=True
)
train.plot_training_curve()

# Store configuration used
config.to_json(filepath_config)
print(f"Created JSON file '{filepath_config}'.")

# Store training curve data
train.save_training_data(filepath_training)
print(f"Created JSON file '{filepath_training}'.")

# Plot training curve
plt.show()
print(train.model.policy)
