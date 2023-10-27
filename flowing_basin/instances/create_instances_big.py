from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from cornflow_client.core.tools import load_json
from itertools import product

# EXAMPLES = ['1', '3']
EXAMPLES = ['Percentile25', 'Percentile75'] + [f'Percentile{i*10:02d}' for i in range(11)]
# NUMS_DAMS = [i for i in range(1, 9)]
NUMS_DAMS = [8, 9, 10]
NUMS_DAYS = [1]

path_historical_data = "../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)

for example, num_dams, num_days in product(EXAMPLES, NUMS_DAMS, NUMS_DAYS):

    base_instance = Instance.from_json(f"instances_base/instance{example}.json")
    start_date, _ = base_instance.get_start_end_datetimes()
    impact_buffer = max(
        [
            base_instance.get_relevant_lags_of_dam(dam_id)[0]
            for dam_index, dam_id in enumerate(base_instance.get_ids_of_dams())
            if dam_index < num_dams
        ]
    )

    length_episode = num_days * 24 * 4 + impact_buffer  # One day (+ impact buffer)
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

    # Assert the instance has num_dams dams
    assert instance.get_num_dams() == num_dams, (
        f"The generated instance does not have {num_dams} dams; it actually has {instance.get_num_dams()}."
    )

    # Assert the instance (until decision horizon) is num_days long
    assert instance.get_decision_horizon() == num_days * 24 * 4, (
        f"The generated instance is not {num_days} days long; "
        f"it is actually {instance.get_decision_horizon() / (24 * 4)} days."
    )

    instance.to_json(f"instances_big/instance{example}_{num_dams}dams_{num_days}days.json")
