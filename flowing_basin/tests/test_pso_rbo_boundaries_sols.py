from flowing_basin.core import Instance, Solution
import matplotlib.pyplot as plt
from itertools import product
import csv

# PSO-RBO plot
# Values of discrete parameters considered (and the way we plot them)
RELVARS = [True, False]
LINES = ['-', '--']
BOUNDARIES = ['periodic', 'nearest', 'intermediate', 'shrink', 'reflective']
COLORS = ['black', 'green', 'red', 'orange', 'blue']

# PSO plot
PSO_COLOR = 'purple'

# MILP plot
MILP_COLOR = 'gray'

# Instances solved
INSTANCES = ['Percentile25', 'Percentile75']
# INSTANCES = ['1', '3']
NUMS_DAMS = [2, 6]

# Other options
PLOT_SOL = False
PLOT_PSO_RBO_SOL = False
PLOT_PSO_SOL = False
PLOT_MILP_SOL = False
SAVE_REPORT = True

report_filepath = "reports/test_pso_rbo_boundaries_sols.csv"

# Create plots with appropriate spacing
fig, axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.7, hspace=0.4)

spacing = 40
report = [
    ['instance'] +
    [f'obj_v={relvar}_b={boundary}' for relvar, boundary in product(RELVARS, BOUNDARIES)]
]
final_objs = dict()
final_objs_norm = dict()
fraction_over_milp = dict()

# Plot the curves
for (instance_index, instance_name), (num_dams_index, num_dams) in product(
    enumerate(INSTANCES), enumerate(NUMS_DAMS)
):

    # Get axes for the given instance and number of dams
    ax = axs[instance_index, num_dams_index]
    min_obj_fun_value = float('inf') if PLOT_PSO_RBO_SOL else 0
    max_obj_fun_value = - float('inf')

    # MILP solution
    sol = Solution.from_json(
        f"../solutions/test_milp/instance{instance_name}_MILP_{num_dams}dams_1days.json"
    )
    milp_obj_fun = sol.get_objective_function()
    if PLOT_MILP_SOL:
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_values()
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=MILP_COLOR, linestyle='-', label='MILP'
        )

    # PSO solution
    sol = Solution.from_json(
        f"../solutions/test_pso/instance{instance_name}_PSO_{num_dams}dams_1days.json"
    )
    if PLOT_PSO_SOL:
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_values()
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=PSO_COLOR, linestyle='-', label='PSO'
        )

    # PSO-RBO solutions
    for (relvar, line), (boundary, color) in product(zip(RELVARS, LINES), zip(BOUNDARIES, COLORS)):

        sol = Solution.from_json(
            f"../solutions/test_pso_rbo_boundaries/instance{instance_name}_PSO-RBO_{num_dams}dams_1days"
            f"_v={relvar}_b={boundary}.json"
        )

        if PLOT_PSO_RBO_SOL:
            time_stamps = sol.get_history_time_stamps()
            obj_fun_values = sol.get_history_values()
            min_obj_fun_value = min(min_obj_fun_value, min(obj_fun_values))
            max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
            ax.plot(
                time_stamps, obj_fun_values, color=color, linestyle=line, label=f'v={relvar}, b={boundary}'
            )

        # Values in report
        final_objs[(instance_name, num_dams, relvar, boundary)] = sol.get_objective_function()
        instance = Instance.from_json(
            f"../instances/instances_big/instance{instance_name}_{num_dams}dams_1days.json"
        )
        avg_inflow = instance.calculate_total_avg_inflow()
        power_installed = sum(instance.get_max_power_of_power_group(dam_id) for dam_id in instance.get_ids_of_dams())
        obj_norm = sol.get_objective_function() / (avg_inflow * power_installed)
        final_objs_norm[(instance_name, num_dams, relvar, boundary)] = obj_norm
        fraction_over_milp[(instance_name, num_dams, relvar, boundary)] = (
            sol.get_objective_function() - milp_obj_fun
        ) / milp_obj_fun

    # Add titles and legends
    if PLOT_SOL:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective function (€)")
        ax.set_ylim(bottom=min_obj_fun_value, top=max_obj_fun_value)
        ax.set_title(f"Instance {instance_name} with {num_dams} dams.")
        ax.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))

# Write report
attributes = [
    (final_objs, "o.f.", "o.f. mean"),
    (final_objs_norm, "norm o.f.", "normalized o.f. mean"),
    (fraction_over_milp, "over MILP", "fraction over MILP mean")
]
for attr, attr_name, aggr_name in attributes:
    for instance_name, num_dams in product(INSTANCES, NUMS_DAMS):
        report.append(
            [f'instance{instance_name} {num_dams}dams ({attr_name})'] +
            [
                round(attr[(instance_name, num_dams, relvar, boundary)], 2)
                for relvar, boundary in product(RELVARS, BOUNDARIES)
            ]
        )
    report.append(
        [aggr_name] +
        [
            round(sum(
                attr[(instance_name, num_dams, relvar, boundary)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
            ) / len([
                attr[(instance_name, num_dams, relvar, boundary)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
            ]), 2)
            for relvar, boundary in product(RELVARS, BOUNDARIES)
        ]
    )

if SAVE_REPORT:
    with open(report_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(report)
    print(f"Created CSV file '{report_filepath}'.")

# Pretty print report
report = [[f"{val:<{spacing}}" if not isinstance(val, float) else f"{val:<{spacing}.2f}" for val in row] for row in report]
print("Report:\n", '\n'.join([''.join(row) for row in report]))

if PLOT_SOL:
    plt.show()

