from flowing_basin.solvers.rl import Training
import pandas as pd
from datetime import datetime
from cornflow_client.core.tools import load_json


START = datetime.strptime("2021-04-03 00:00", "%Y-%m-%d %H:%M")
PATH_TRAINING_DATA = "../data/rl_training_data/training_data.pickle"
TRAINING_DATA = pd.read_pickle(PATH_TRAINING_DATA)
EXAMPLE_NUMBER = 1

for num_dams in [4]:

    for num_days in [1]:

        length_episode = num_days * 24 * 4 + 3  # One day (+ impact buffer)
        path_constants = f"../data/rl_training_data/constants_{num_dams}dams.json"
        instance = Training.create_instance(
            length_episodes=length_episode,
            constants=load_json(path_constants),
            training_data=TRAINING_DATA,
            initial_row=START,
        )

        inconsistencies = instance.check()
        if inconsistencies:
            raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

        instance.to_json(f"../data/input_example{EXAMPLE_NUMBER}_{num_dams}dams_{num_days}days.json")
