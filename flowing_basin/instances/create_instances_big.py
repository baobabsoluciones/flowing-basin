from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from datetime import datetime
from cornflow_client.core.tools import load_json


START = datetime.strptime("2021-04-03 00:00", "%Y-%m-%d %H:%M")
EXAMPLE_NUMBER = 1

path_historical_data = "../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)

for num_dams in [1]:

    for num_days in [1]:

        length_episode = num_days * 24 * 4 + 3  # One day (+ impact buffer)
        path_constants = f"../data/constants/constants_{num_dams}dams.json"
        instance = RLEnvironment.create_instance(
            length_episodes=length_episode,
            constants=load_json(path_constants),
            historical_data=historical_data,
            initial_row=START,
        )

        inconsistencies = instance.check()
        if inconsistencies:
            raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

        instance.to_json(f"instances_big/instance{EXAMPLE_NUMBER}_{num_dams}dams_{num_days}days.json")
