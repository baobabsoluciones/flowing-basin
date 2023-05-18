from flowing_basin.solvers.rl import RLConfiguration, RLTrain
from datetime import datetime

config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    flow_smoothing=2,
    num_prices=16,
    num_unreg_flows=16,
    num_incoming_flows=16,
    length_episodes=24 * 4 + 3,
    fast_mode=True,
)
train = RLTrain(
    config=config,
    path_constants="../data/rl_training_data/constants.json",
    path_training_data="../data/rl_training_data/training_data.pickle",
)
train.solve(num_episodes=100, path_agent=f"../data/RL_model_{datetime.now().strftime('%Y-%m-%d %H.%M')}.zip")
