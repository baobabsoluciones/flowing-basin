from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime
from matplotlib import pyplot as plt

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
        "future_inflows", "past_groups", "past_powers", "past_clipped", "past_periods"
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
    num_episodes=10,
    path_agent=f"../solutions/RL_model_{datetime.now().strftime('%Y-%m-%d %H.%M')}.zip",
    periodic_evaluation=True
)
train.plot_training_curve()
plt.show()
print(train.model.policy)
