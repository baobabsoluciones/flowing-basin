from flowing_basin.core import Solution
import matplotlib.pyplot as plt
from itertools import product

# Values of discrete parameters to consider (and the way we plot them)
RELVARS_LINES = {
    True: '-',
    False: '--'
}
BOUNDARIES_COLORS = {
    'periodic': 'black',
    'nearest': 'green',
    'intermediate': 'red',
    'shrink': 'orange',
    'reflective': 'blue'
}

# Instances to solve
INSTANCES = ['Percentile25', 'Percentile75']
NUMS_DAMS = [2, 6]

# Create plots with appropriate spacing
fig, axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.6, hspace=0.15)

# Plot the curves
for (instance_index, instance_name), (num_dams_index, num_dams), (relvar, line), (boundary, color) in product(
    enumerate(INSTANCES), enumerate(NUMS_DAMS), RELVARS_LINES.items(), BOUNDARIES_COLORS.items()
):
    path_sol = (
        f"../solutions/test_pso_rbo_boundaries/instance{instance_name}_PSO-RBO_{num_dams}dams_1days"
        f"_v={relvar}_b={boundary}.json"
    )
    sol = Solution.from_json(path_sol)

    time_stamps = sol.get_history_time_stamps()
    obj_fun_values = sol.get_history_values()

    axs[instance_index, num_dams_index].plot(
        time_stamps, obj_fun_values, color=color, linestyle=line, label=f'v={relvar}, b={boundary}'
    )

# Add titles and legends
for (instance_index, instance_name), (num_dams_index, num_dams) in product(
    enumerate(INSTANCES), enumerate(NUMS_DAMS)
):
    ax = axs[instance_index, num_dams_index]
    ax.set_title(f"Instance {instance_name} with {num_dams} dams.")
    ax.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

