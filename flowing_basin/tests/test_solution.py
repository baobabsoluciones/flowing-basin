from flowing_basin.core import Instance, Solution, Configuration
from flowing_basin.tools import RiverBasin
from matplotlib import pyplot as plt


PLOT_SOL = True

# INSTANCE = 1
# NUM_DAMS = 2
# instance = Instance.from_json(f"../instances/instances_big/instance{INSTANCE}_{NUM_DAMS}dams_1days.json")
instance = Instance.from_json(f"../instances/instances_big/instance1_2dams_1days.json")

# SOLVER = "Heuristic"
# SOL_DATETIME = "2023-09-25_14-07"
# solution = Solution.from_json(f"../solutions/instance{INSTANCE}_{SOLVER}_{NUM_DAMS}dams_1days_time{SOL_DATETIME}.json")
solution = Solution.from_json("../solutions/test_pso/instance1_PSO_2dams_1days.json")

# Make sure data follows schema and has no inconsistencies
inconsistencies = solution.check()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")
assert solution.complies_with_flow_smoothing(
    flow_smoothing=2,
    initial_flows={
        dam_id: instance.get_initial_lags_of_channel(dam_id)[0]
        for dam_id in instance.get_ids_of_dams()
    }
)
    
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
    print(f"{dam_id} predicted flows:", solution.get_predicted_exiting_flows_of_dam(dam_id))
    print(f"{dam_id} volumes:", solution.get_volumes_of_dam(dam_id))
    print(f"{dam_id} powers:", solution.get_powers_of_dam(dam_id))
print("Prices:", solution.get_all_prices())

# History values
time_stamps = solution.get_history_time_stamps()
obj_values = solution.get_history_values()
if time_stamps is not None and obj_values is not None:
    print(' '.join([f"{el:^15}" for el in time_stamps]))
    print(' '.join([f"{el:^15.2f}" for el in obj_values]))

# Plot solution
if PLOT_SOL:
    for dam_id in solution.get_ids_of_dams():
        fig, ax = plt.subplots()
        solution.plot_solution_for_dam(dam_id, ax)
        plt.show()

# Check with river basin simulator
river_basin = RiverBasin(instance=instance, mode="linear")
flows = solution.get_flows_array()
river_basin.deep_update_flows(flows)

# Compare flows and actual flows
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
            
# Print simulation historic values with solution
income_energy = 0.
startups = 0
limits = 0
print(river_basin.history.to_string())
for dam_id in instance.get_ids_of_dams():
    income_energy += river_basin.history[f'{dam_id}_income'].sum()
    print(f"{dam_id}'s income from energy: {income_energy}")
    startups += river_basin.history[f'{dam_id}_startups'].sum()
    print(f"{dam_id}'s number of startups: {startups}")
    limits += river_basin.history[f'{dam_id}_limits'].sum()
    print(f"{dam_id}'s number of limit zones: {limits}")

# Calculate obj fun manually
config = solution.get_configuration()
if config is not None:
    config = Configuration.from_dict(config)
    obj_fun_calculated = income_energy - startups * config.startups_penalty - limits * config.limit_zones_penalty
    print("Calculated objective function:", obj_fun_calculated)
    obj_fun_stored = solution.get_objective_function()
    if obj_fun_stored is not None:
        assert abs(obj_fun_calculated - obj_fun_stored) < 1e-3

