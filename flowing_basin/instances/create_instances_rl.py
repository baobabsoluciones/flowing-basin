from flowing_basin.core import Instance
from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd

EXAMPLES = [
    f"Percentile{percentile:02}"
    for percentile in range(0, 110, 10)
]
NUM_STEPS_LOOKAHEAD = 16

path_constants = "../data/constants/constants_2dams.json"
constants = load_json(path_constants)

path_historical_data = "../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)

for example in EXAMPLES:

    base_instance = Instance.from_json(f"instances_base/instance{example}.json")
    start_date = base_instance.get_start_decisions_datetime()

    # Create instance
    instance = RLEnvironment.create_instance(
        length_episodes=24 * 4 + 3,
        constants=constants,
        historical_data=historical_data,
        info_buffer_start=NUM_STEPS_LOOKAHEAD,
        info_buffer_end=NUM_STEPS_LOOKAHEAD,
        initial_row_decisions=start_date,
        instance_name=example
    )

    # Check instance
    inconsistencies = instance.check()
    if inconsistencies:
        raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

    # Save instance
    instance.to_json(f"instances_rl/instance{example}_expanded{NUM_STEPS_LOOKAHEAD}steps_backforth.json")
