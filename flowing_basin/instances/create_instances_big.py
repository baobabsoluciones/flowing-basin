from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from datetime import datetime
from cornflow_client.core.tools import load_json


EXAMPLE_NUMBER = 3
start_dates = {
    1: "2021-04-03 00:00",
    3: "2020-12-01 05:45"
}
start_date = datetime.strptime(start_dates[EXAMPLE_NUMBER], "%Y-%m-%d %H:%M")

path_historical_data = "../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)

for num_dams in [2]:

    for num_days in [1]:

        length_episode = num_days * 24 * 4 + 3  # One day (+ impact buffer)
        path_constants = f"../data/constants/constants_{num_dams}dams.json"
        instance = RLEnvironment.create_instance(
            length_episodes=length_episode,
            constants=load_json(path_constants),
            historical_data=historical_data,
            initial_row=start_date,
        )

        inconsistencies = instance.check()
        if inconsistencies:
            raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

        instance.to_json(f"instances_big/instance{EXAMPLE_NUMBER}_{num_dams}dams_{num_days}days.json")
