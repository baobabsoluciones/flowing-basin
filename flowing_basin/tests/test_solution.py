from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin
import numpy as np

INSTANCE = 1
NUM_DAMS = 2

solution = Solution.from_json(f"../solutions/instance{INSTANCE}_LPmodel_{NUM_DAMS}dams_1days_time2023-09-05_13-39.json")
# solution = Solution.from_json(f"../solutions/instance{INSTANCE}_Heuristic_{NUM_DAMS}dams_1days_time2023-09-07_22-11.json")

# Make sure data follows schema and has no inconsistencies
inconsistencies = solution.check()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

# Print dam info
for dam_id in solution.get_ids_of_dams():
    print(f"{dam_id} flows:", solution.get_exiting_flows_of_dam(dam_id))
    print(f"{dam_id} volumes:", solution.get_volumes_of_dam(dam_id))
    print(f"{dam_id} powers:", solution.get_powers_of_dam(dam_id))

# Print general info
print("Objective function (€):", solution.get_objective_function())

# Check with river basin simulator
instance = Instance.from_json(f"../instances/instances_big/instance{INSTANCE}_{NUM_DAMS}dams_1days.json")
river_basin = RiverBasin(instance=instance, mode="linear")
flows = solution.get_exiting_flows_array()
river_basin.deep_update_flows(flows)
actual_flows = river_basin.all_past_clipped_flows
print("Flows array:", flows.tolist())
print("Actual flows:", actual_flows.tolist())
for dam_index, dam_id in enumerate(instance.get_ids_of_dams()):
    for time_step in range(instance.get_decision_horizon()):
        flow = flows[time_step, dam_index]
        actual_flow = actual_flows[time_step, dam_index]
        if abs(flow - actual_flow) > 1e-6:
            print(
                f"WARNING - For dam {dam_id} and time {time_step}, "
                f"the flow is {flow} but the actual flow is {actual_flow}"
            )
