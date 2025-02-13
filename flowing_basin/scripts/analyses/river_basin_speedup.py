"""
river_basin_speedup.py
This script analyzes the speedup for using NumPy arrays instead of an inner loop in the simulator
"""

from flowing_basin.tools import RiverBasin
from flowing_basin.core import Instance
from flowing_basin.solvers.common import barchart_instances, extract_number, get_episode_length, CONSTANTS_PATH, HISTORICAL_DATA_PATH
from flowing_basin.solvers.rl import RLEnvironment
from cornflow_client.core.tools import load_json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

NUM_REPLICATIONS = 5
CASES = [f'{num_sols} scenarios' for num_sols in [10, 20, 50, 100]]
METHOD_SLOW = "Inner loop"
METHOD_FAST = "Array computation"
NUM_DAMS = 6


if __name__ == "__main__":

    # Colors
    set1_cmap = plt.get_cmap('Set1')
    method_colors = {METHOD_SLOW: set1_cmap(0), METHOD_FAST: set1_cmap(1)}

    # We need to build a dict[method, dict[case, values]]
    values = {METHOD_SLOW: dict(), METHOD_FAST: dict()}

    # Slow method
    for case in CASES:

        values[METHOD_SLOW][case] = []
        values[METHOD_FAST][case] = []
        num_sols = extract_number(case)

        for replication in range(NUM_REPLICATIONS):

            # Create a random instance
            constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=NUM_DAMS)))
            historical_data = pd.read_pickle(HISTORICAL_DATA_PATH)
            instance = RLEnvironment.create_instance(
                length_episodes=get_episode_length(constants=constants), constants=constants, historical_data=historical_data
            )

            # Generate random flows between 0 and max_flow
            flows = np.random.rand(
                instance.get_largest_impact_horizon(), instance.get_num_dams(), num_sols
            )
            flows = flows * np.array(
                [instance.get_max_flow_of_channel(dam_id) for dam_id in instance.get_ids_of_dams()]
            ).reshape((1, -1, 1))

            # Execute these flows - SLOW METHOD
            start = perf_counter()
            transposed_flows = flows.transpose(2, 0, 1)
            river_basin_slow = RiverBasin(instance=instance, num_scenarios=1, do_history_updates=False, mode="linear")
            for single_flows in transposed_flows:
                river_basin_slow.deep_update_flows(single_flows.reshape(-1, NUM_DAMS, 1))
                river_basin_slow.reset()
            exec_time = perf_counter() - start
            print(f"[Case {case}] [Replication {replication}] Executed slow method in {exec_time}s.")
            values[METHOD_SLOW][case].append(exec_time)

            # Execute these flows - FAST METHOD
            start = perf_counter()
            river_basin_fast = RiverBasin(instance=instance, num_scenarios=num_sols, do_history_updates=False, mode="linear")
            river_basin_fast.deep_update_flows(flows)
            exec_time = perf_counter() - start
            print(f"[Case {case}] [Replication {replication}] Executed fast method in {exec_time}s.")
            values[METHOD_FAST][case].append(exec_time)

    barchart_instances(
        filename="river_basin_speedup/barchart.eps", values=values, vertical_x_labels=False,
        y_label="Execution time (s)", x_label=f"Number of scenarios (with {NUM_DAMS} dams)",
        full_title="Speedup of array computation", solver_colors=method_colors
    )
