from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime
from matplotlib import pyplot as plt
from dataclasses import asdict
import json

current_datetime = datetime.now().strftime('%Y-%m-%d %H.%M')
filepath_agent = f"../solutions/rl_models/RL_model_{current_datetime}.zip"
filepath_config = f"../solutions/rl_models/RL_model_{current_datetime}_config.json"

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
    num_steps_sight=16,
    length_episodes=24 * 4 + 3,
    log_ep_freq=5,
    eval_ep_freq=5,
    eval_num_episodes=10,
    do_history_updates=False,
)
train = RLTrain(
    config=config,
    path_constants="../data/constants/constants_2dams.json",
    path_train_data="../data/history/historical_data_clean_train.pickle",
    path_test_data="../data/history/historical_data_clean_test.pickle"
)

train.solve(
    num_episodes=200,
    path_agent=filepath_agent,
    periodic_evaluation=True
)
train.plot_training_curve()

# Store configuration used
with open(filepath_config, 'w') as file:
    json.dump(asdict(config), file, indent=2)
print(f"Created JSON file '{filepath_config}'.")

plt.show()
print(train.model.policy)
