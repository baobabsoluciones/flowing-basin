"""
create_instances_big.py
This script creates versions of the instances in `instances_base`
that have different numbers of dams
"""

from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLEnvironment
from flowing_basin.solvers.common import get_episode_length, CONSTANTS_PATH
import pandas as pd
from cornflow_client.core.tools import load_json
from itertools import product

if __name__ == "__main__":

    EXAMPLES = [f'Percentile{i*10:02d}' for i in range(11)] + ['Percentile25', 'Percentile75']
    NUMS_DAMS = [1]
    NUMS_DAYS = [1]

    path_historical_data = "../data/history/historical_data.pickle"
    historical_data = pd.read_pickle(path_historical_data)

    for example, num_dams, num_days in product(EXAMPLES, NUMS_DAMS, NUMS_DAYS):

        base_instance = Instance.from_json(f"instances_base/instance{example}.json")
        start_date = base_instance.get_start_decisions_datetime()

        constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=num_dams)))
        length_episode = get_episode_length(constants=constants, num_days=num_days)
        instance = RLEnvironment.create_instance(
            length_episodes=length_episode,
            constants=constants,
            historical_data=historical_data,
            initial_row_decisions=start_date,
            instance_name=example
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
