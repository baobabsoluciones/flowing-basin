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
# RELVARS = [False]
# LINES = ['--']
# BOUNDARIES = ['intermediate']
# COLORS = ['red']

# PSO plot
PSO_COLOR = 'purple'

# MILP plot
MILP_COLOR = 'gray'

# Instances solved
INSTANCES = ['Percentile25', 'Percentile75']
# INSTANCES = ['1', '3']
NUMS_DAMS = [2, 6]
# NUMS_DAMS = [6, 8, 10, 12]
VOL_BONUS = False
POWER_PENALTY = True

# Other options
PLOT_SOL = True
PLOT_PSO_RBO_SOL = True
PLOT_PSO_SOL = True
PLOT_MILP_SOL = True
SAVE_REPORT = True
TIME_LIMITS = [5 * 60, 15 * 60]
OBJ_FUN_NORM_METHOD = 'NEW'

report_filepath = (
    f"reports/test_pso_rbo_boundaries_sols_{OBJ_FUN_NORM_METHOD}"
    f"{'_VolExceed' if VOL_BONUS else ''}{'_NoPowerPenalty' if not POWER_PENALTY else ''}.csv"
)

# Create plots with appropriate spacing
fig, axs = plt.subplots(len(INSTANCES), len(NUMS_DAMS))
plt.subplots_adjust(wspace=0.7, hspace=0.4)

# First row of report
first_row = ['instance']
for time_limit in TIME_LIMITS:
    first_row += [f'MILP gap ({time_limit}s)', f'PSO ({time_limit}s)']
    for relvar, boundary in product(RELVARS, BOUNDARIES):
        first_row += [f'obj_v={relvar}_b={boundary} ({time_limit}s)']
report = [first_row]

# Values that will appear in report
spacing = 40  # When printing the report
final_objs = dict()
final_objs_norm = dict()
fraction_over_milp = dict()
fraction_over_milp_pso = dict()
milp_obj_fun = dict()
milp_final_gaps = dict()

# Plot the curves
for (instance_index, instance_name), (num_dams_index, num_dams) in product(
    enumerate(INSTANCES), enumerate(NUMS_DAMS)
):

    # Get axes for the given instance and number of dams
    ax = axs[instance_index, num_dams_index]
    min_obj_fun_value = 0  # float('inf')
    max_obj_fun_value = - float('inf')

    # MILP solution
    sol = Solution.from_json(
        f"../solutions/test_milp/instance{instance_name}_MILP_{num_dams}dams_1days"
        f"{'_VolExceed' if VOL_BONUS else ''}{'_NoPowerPenalty' if not POWER_PENALTY else ''}.json"
    )
    for time_limit in TIME_LIMITS:
        milp_obj_fun[(instance_name, num_dams, time_limit)] = sol.get_history_objective_function_value(time_limit)
        milp_final_gaps[(instance_name, num_dams, time_limit)] = sol.get_history_gap_value(time_limit)
    if PLOT_MILP_SOL:
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_objective_function_values()
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=MILP_COLOR, linestyle='-', label='MILP'
        )

    # PSO solution
    sol = Solution.from_json(
        f"../solutions/test_pso/instance{instance_name}_PSO_{num_dams}dams_1days"
        f"{'_VolExceed' if VOL_BONUS else ''}{'_NoPowerPenalty' if not POWER_PENALTY else ''}.json"
    )
    if PLOT_PSO_SOL:
        time_stamps = sol.get_history_time_stamps()
        obj_fun_values = sol.get_history_objective_function_values()
        # min_obj_fun_value = min(min_obj_fun_value, min(obj_fun_values))
        max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
        ax.plot(
            time_stamps, obj_fun_values, color=PSO_COLOR, linestyle='-', label='PSO'
        )
        for time_limit in TIME_LIMITS:
            pso_obj_fun = sol.get_history_objective_function_value(time_limit)
            fraction_over_milp_pso[(instance_name, num_dams, time_limit)] = (
                pso_obj_fun - milp_obj_fun[(instance_name, num_dams, time_limit)]
            ) / milp_obj_fun[(instance_name, num_dams, time_limit)] if milp_obj_fun[(instance_name, num_dams, time_limit)] > 0 else (
                float('inf')
            )

    # PSO-RBO solutions
    for (relvar, line), (boundary, color) in product(zip(RELVARS, LINES), zip(BOUNDARIES, COLORS)):

        sol = Solution.from_json(
            f"../solutions/test_pso_rbo_boundaries/instance{instance_name}_PSO-RBO_{num_dams}dams_1days"
            f"_v={relvar}_b={boundary}"
            f"{'_VolExceed' if VOL_BONUS else ''}{'_NoPowerPenalty' if not POWER_PENALTY else ''}.json"
        )

        if PLOT_PSO_RBO_SOL:
            time_stamps = sol.get_history_time_stamps()
            obj_fun_values = sol.get_history_objective_function_values()
            min_obj_fun_value = min(min_obj_fun_value, min(obj_fun_values))
            max_obj_fun_value = max(max_obj_fun_value, max(obj_fun_values))
            ax.plot(
                time_stamps, obj_fun_values, color=color, linestyle=line, label=f'v={relvar}, b={boundary}'
            )

        # Values in report
        for time_limit in TIME_LIMITS:

            pso_rbo_obj_fun = sol.get_history_objective_function_value(time_limit)
            final_objs[(instance_name, num_dams, relvar, boundary, time_limit)] = pso_rbo_obj_fun

            instance = Instance.from_json(
                f"../instances/instances_big/instance{instance_name}_{num_dams}dams_1days.json"
            )
            avg_inflow = instance.get_total_avg_inflow()
            power_installed = sum(instance.get_max_power_of_power_group(dam_id) for dam_id in instance.get_ids_of_dams())
            if OBJ_FUN_NORM_METHOD == 'NEW':
                avg_price = instance.get_avg_price()
                obj_norm = pso_rbo_obj_fun / (avg_inflow * avg_price)
            else:
                obj_norm = pso_rbo_obj_fun / (avg_inflow * power_installed)
            final_objs_norm[(instance_name, num_dams, relvar, boundary, time_limit)] = obj_norm

            fraction_over_milp[(instance_name, num_dams, relvar, boundary, time_limit)] = (
                pso_rbo_obj_fun - milp_obj_fun[(instance_name, num_dams, time_limit)]
            ) / milp_obj_fun[(instance_name, num_dams, time_limit)] if milp_obj_fun[(instance_name, num_dams, time_limit)] > 0 else (
                float('inf')
            )

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

    # Values for each instance
    for instance_name, num_dams in product(INSTANCES, NUMS_DAMS):
        row = [f'instance{instance_name} {num_dams}dams ({attr_name})']
        for time_limit in TIME_LIMITS:
            row += [milp_final_gaps[(instance_name, num_dams, time_limit)], fraction_over_milp_pso[(instance_name, num_dams, time_limit)]]
            for relvar, boundary in product(RELVARS, BOUNDARIES):
                row += [round(attr[(instance_name, num_dams, relvar, boundary, time_limit)], 2)]
        report.append(row)

    # Mean across all instances
    final_row = [aggr_name]
    for time_limit in TIME_LIMITS:
        final_row += [
            sum(
                milp_final_gaps[(instance_name, num_dams, time_limit)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
            ) / len([
                milp_final_gaps[(instance_name, num_dams, time_limit)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
            ]),
            sum(
                fraction_over_milp_pso[(instance_name, num_dams, time_limit)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
                if fraction_over_milp_pso[(instance_name, num_dams, time_limit)] < float('inf')
            ) / len([
                fraction_over_milp_pso[(instance_name, num_dams, time_limit)]
                for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
                if fraction_over_milp_pso[(instance_name, num_dams, time_limit)] < float('inf')
            ])
        ]
        for relvar, boundary in product(RELVARS, BOUNDARIES):
            final_row += [
                round(sum(
                    attr[(instance_name, num_dams, relvar, boundary, time_limit)]
                    for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
                    if attr[(instance_name, num_dams, relvar, boundary, time_limit)] < float('inf')
                ) / len([
                    attr[(instance_name, num_dams, relvar, boundary, time_limit)]
                    for instance_name, num_dams in product(INSTANCES, NUMS_DAMS)
                    if attr[(instance_name, num_dams, relvar, boundary, time_limit)] < float('inf')
                ]), 2)
            ]
    report.append(final_row)

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

