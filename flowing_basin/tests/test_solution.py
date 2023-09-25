from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin
from matplotlib import pyplot as plt

INSTANCE = 1
NUM_DAMS = 2
SOLVER = "PSO"
SOL_DATETIME = "2023-09-25_13-20"
PLOT_SOL = True

solution = Solution.from_json(f"../solutions/instance{INSTANCE}_{SOLVER}_{NUM_DAMS}dams_1days_time{SOL_DATETIME}.json")

# Make sure data follows schema and has no inconsistencies
inconsistencies = solution.check()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

# Print general info
instance_datetimes = solution.get_instance_start_end_datetimes()
if instance_datetimes is not None:
    print(
        f"The instance solved starts at {instance_datetimes[0].strftime('%Y-%m-%d %H:%M')} "
        f"and ends at {instance_datetimes[1].strftime('%Y-%m-%d %H:%M')}."
    )
solution_datetime = solution.get_solution_datetime()
if solution_datetime is not None:
    print(
        f"The solution was obtained at {solution_datetime.strftime('%Y-%m-%d %H:%M')} "
        f"using solver {solution.get_solver()}."
    )
print("Objective function (€):", solution.get_objective_function())

# Print dam info
for dam_id in solution.get_ids_of_dams():
    print(f"{dam_id} flows:", solution.get_exiting_flows_of_dam(dam_id))
    print(f"{dam_id} volumes:", solution.get_volumes_of_dam(dam_id))
    print(f"{dam_id} powers:", solution.get_powers_of_dam(dam_id))
print("Prices:", solution.get_all_prices())

# Plot solution
if PLOT_SOL:
    for dam_id in solution.get_ids_of_dams():
        fig, ax = plt.subplots()
        solution.plot_solution_for_dam(dam_id, ax)
        plt.show()

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
