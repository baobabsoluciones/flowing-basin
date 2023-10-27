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
# INSTANCES = ['Percentile25', 'Percentile75']
INSTANCES = ['1', '3']
NUMS_DAMS = [2, 6]

# Other options
PLOT_SOL = True
PLOT_PSO_RBO_SOL = False
PLOT_PSO_SOL = True
PLOT_MILP_SOL = True
SAVE_REPORT = False

report_filepath = "reports/test_pso_rbo_boundaries_sols.csv"

# Create plots with appropriate spacing
fig, axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.7, hspace=0.4)

spacing = 25
report = [
    ['instance'] +
    [f'obj_v={relvar}_b={boundary}' for relvar, boundary in product(RELVARS, BOUNDARIES)]
]
final_objs = dict()
avg_final_objs = {(relvar, boundary): 0. for relvar, boundary in product(RELVARS, BOUNDARIES)}

# Plot the curves
for (instance_index, instance_name), (num_dams_index, num_dams) in product(
    enumerate(INSTANCES), enumerate(NUMS_DAMS)
):

    # Get axes for the given instance and number of dams
    ax = axs[instance_index, num_dams_index]
    min_obj_fun_value = float('inf') if PLOT_PSO_RBO_SOL else 0
    max_obj_fun_value = - float('inf')

    # MILP solution
    if PLOT_MILP_SOL:
        sol = Solution.from_json(
            f"../solutions/test_milp/instance{instance_name}_MILP_{num_dams}dams_1days.json"
        )
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_values()
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=MILP_COLOR, linestyle='-', label='MILP'
        )

    # PSO solution
    if PLOT_PSO_SOL:
        sol = Solution.from_json(
            f"../solutions/test_pso/instance{instance_name}_PSO_{num_dams}dams_1days.json"
        )
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_values()
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=PSO_COLOR, linestyle='-', label='PSO'
        )

    # PSO-RBO solutions
    if PLOT_PSO_RBO_SOL:
        for (relvar, line), (boundary, color) in product(zip(RELVARS, LINES), zip(BOUNDARIES, COLORS)):

            sol = Solution.from_json(
                f"../solutions/test_pso_rbo_boundaries/instance{instance_name}_PSO-RBO_{num_dams}dams_1days"
                f"_v={relvar}_b={boundary}.json"
            )

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
            avg_final_objs[(relvar, boundary)] += sol.get_objective_function() / (avg_inflow * power_installed)

        # Add row in report about the current instance
        report.append(
            [f'instance{instance_name}_{num_dams}dams'] +
            [
                round(final_objs[(instance_name, num_dams, relvar, boundary)], 2)
                for relvar, boundary in product(RELVARS, BOUNDARIES)
            ]
        )

    # Add titles and legends
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Objective function (€)")
    ax.set_ylim(bottom=min_obj_fun_value, top=max_obj_fun_value)
    ax.set_title(f"Instance {instance_name} with {num_dams} dams.")
    ax.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))

# Add row in report about the current instance
report.append(
    ['avg_normalized'] +
    [
        round(avg_final_objs[(relvar, boundary)], 2)
        for relvar, boundary in product(RELVARS, BOUNDARIES)
    ]
)

if SAVE_REPORT:
    with open(report_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(report)
    print(f"Created CSV file '{report_filepath}'.")

# Pretty print report
report = [[f"{val:^{spacing}}" if not isinstance(val, float) else f"{val:^{spacing}.2f}" for val in row] for row in report]
print("Report:\n", '\n'.join([''.join(row) for row in report]))

if PLOT_SOL:
    plt.show()

