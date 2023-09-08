from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from cornflow_client.core.tools import load_json
from itertools import product

# EXAMPLES = ['1']
EXAMPLES = [f'_intermediate{i}' for i in range(11)]
NUMS_DAMS = [2]
NUMS_DAYS = [1]

path_historical_data = "../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)

for example, num_dams, num_days in product(EXAMPLES, NUMS_DAMS, NUMS_DAYS):

    start_date, _ = Instance.from_json(f"instances_base/instance{example}.json").get_start_end_datetimes()

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

    instance.to_json(f"instances_big/instance{example}_{num_dams}dams_{num_days}days.json")
