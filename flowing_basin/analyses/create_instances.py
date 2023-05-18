from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
import pandas as pd
from datetime import datetime

NUM_STEPS_LOOKAHEAD = 16

path_constants = "../data/rl_training_data/constants.json"
constants = load_json(path_constants)

path_training_data = "../data/rl_training_data/training_data.pickle"
training_data = pd.read_pickle(path_training_data)

config = RLConfiguration(
    startups_penalty=50,
    limit_zones_penalty=50,
    mode="linear",
    flow_smoothing=2,
    num_prices=NUM_STEPS_LOOKAHEAD,
    num_unreg_flows=NUM_STEPS_LOOKAHEAD,
    num_incoming_flows=NUM_STEPS_LOOKAHEAD,
    length_episodes=24 * 4 + 3,
)

instance = RLEnvironment.create_instance(
    length_episodes=24 * 4 + 3,
    constants=constants,
    training_data=training_data,
    config=config,
    initial_row=datetime.strptime("2021-04-03 00:00", "%Y-%m-%d %H:%M"),
)
instance.to_json(f"../data/input_example1_expanded{NUM_STEPS_LOOKAHEAD}steps.json")
