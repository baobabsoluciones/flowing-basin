"""
plot_power_group.py
This script draws the power curve of both dams,
optionally adding a histogram of the outflows of a solver
"""

from cornflow_client.core.tools import load_json
from flowing_basin.tools import PowerGroup
from flowing_basin.core import Instance
from flowing_basin.core.utils import lighten_color
from flowing_basin.solvers.rl import ReinforcementLearning, GeneralConfiguration
from flowing_basin.solvers.common import CONSTANTS_PATH
from flowing_basin.solvers import Baseline, Baselines
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import numpy as np


PLOT_POWER_CURVE = True
PLOT_SOLVER_FLOWS = False
USE_TRANSPARENCY = False  # This does not allow saving in .eps format
PUT_TEXT = True
SET_YLIM = False
DAM_IDS = ["dam1", "dam2"]  # Put None to plot all dams, or [dam_id] for a single dam
SOLVER = "PSO (general)"  # Any solver (or agent) with solutions in rl_data/baselines, such as "MILP" or "PSO (general)"
GENERAL = 'G2'  # General configuration from which to get the solutions when PLOT_SOLVER_FLOWS = True
FILENAME = f'plot_power_group/fig_power_vs_turbine_flow'
# DAM_NAMES = {}  # DAM_NAMES = {'dam1': 'first subsystem', 'dam2': 'second subsystem'}
DAM_NAMES = {'dam1': 'first subsystem', 'dam2': 'second subsystem'}

# Define the name of the filename and the title of the plot
if PLOT_SOLVER_FLOWS:
    filename_solver = f'_vs_{SOLVER}_outflows_{GENERAL}'
    plot_title = f' with {SOLVER} outflows in {GENERAL}'
    if not USE_TRANSPARENCY:
        filename_solver += f"_no_transparent"
else:
    filename_solver = ''
    plot_title = ''

# Define the IDs of the dams to plot
general_config_dict = Baseline.get_general_config_dict(GENERAL)
general_config_obj = GeneralConfiguration.from_dict(general_config_dict)
num_dams = general_config_obj.num_dams
constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=num_dams)))
if DAM_IDS is None:
    DAM_IDS = constants.get_ids_of_dams()
    filename_dams = ''
else:
    assert isinstance(DAM_IDS, list)
    filename_dams = '_' + '_'.join(DAM_IDS)
num_dams_plot = len(DAM_IDS)

# Define the complete filename
filename = FILENAME + filename_solver + filename_dams
if not PLOT_POWER_CURVE:
    filename += "_no_power_curve"

# Get the data for the plot
if PLOT_SOLVER_FLOWS:
    solver_flows = {dam_id: [] for dam_id in constants.get_ids_of_dams()}
    baselines = Baselines(general_config=GENERAL, solvers=[SOLVER])
    # If there are multiple replications, get the best solution for each instance
    instances = {sol.get_instance_name() for sol in baselines.solutions}
    sols = [
        max(
            [sol for sol in baselines.solutions if sol.get_instance_name() == instance],
            key=lambda s: s.get_objective_function()
        )
        for instance in instances
    ]
    for sol in sols:
        for dam_id in sol.get_ids_of_dams():
            solver_flows[dam_id].extend(sol.get_exiting_flows_of_dam(dam_id))
else:
    solver_flows = None

# Define the layout for the graph
if num_dams_plot < 4:
    layout = dict(nrows=1, ncols=num_dams_plot, figsize=(6 * num_dams_plot, 5))
else:
    num_cols = ceil(sqrt(num_dams_plot))
    num_rows = ceil(num_dams_plot / num_cols)
    layout = dict(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 6 * num_rows))
fig, axs = plt.subplots(**layout)

for i, dam_id in enumerate(DAM_IDS):

    data = constants.get_turbined_flow_obs_for_power_group(dam_id)
    observed_flows = data['observed_flows']
    observed_powers = data['observed_powers']
    max_flow = max(observed_flows)

    startup_flows = constants.get_startup_flows_of_power_group(dam_id)
    shutdown_flows = constants.get_shutdown_flows_of_power_group(dam_id)

    flow_bins = PowerGroup.get_turbined_bins_and_groups(startup_flows, shutdown_flows)
    print(dam_id, "flow bins:", flow_bins)

    # Get the axis
    if num_dams_plot == 1:
        ax = axs
    elif num_dams_plot < 4:
        ax = axs[i]
    else:
        col = i % layout["ncols"]
        row = floor(i / layout["ncols"])
        ax = axs[row, col]

    flows, groups = flow_bins
    i = 0
    while i < len(flows):

        if PLOT_POWER_CURVE:
            # Shaded area
            limits = (flows[i], flows[i + 1]) if i < len(flows) - 1 else (flows[i], max_flow)
            x = np.linspace(limits[0], limits[1])
            y = np.interp(x=x, xp=observed_flows, fp=observed_powers)
            col = 'lightgreen' if i % 2 == 0 else 'lightcoral'
            ax.fill_between(x, y, facecolor=lighten_color(col))

            if PUT_TEXT:
                # Text
                num_groups = groups[i + 1].item()
                darkened_color = 'maroon' if col == 'lightcoral' else 'darkgreen'
                ax.text(
                    (limits[0] + limits[1]) / 2, y.mean() / 2,
                    f"{int(num_groups) if num_groups.is_integer() else num_groups} turbines", ha='center', va='bottom',
                    fontsize=12, color=darkened_color
                )

        i += 1

    if PLOT_SOLVER_FLOWS:
        flows_limits = flow_bins[0]
        flows_limits = np.append(0, np.append(flows_limits, max_flow))
        bins = []
        for j in range(len(flows_limits) - 1):
            bins.extend(np.linspace(flows_limits[j], flows_limits[j + 1], 6, endpoint=False))
        bins.append(flows_limits[-1])
        print(dam_id, "bins:", bins)
        twin_ax = ax.twinx()
        hist_kwargs = dict(
            x=solver_flows[dam_id], bins=bins, label=f"{SOLVER} outflows"
        )
        if USE_TRANSPARENCY:
            hist_kwargs.update(color='orange', alpha=0.5)  # noqa
        else:
            hist_kwargs.update(color=lighten_color('green'))
        twin_ax.hist(**hist_kwargs)
        twin_ax.legend()
        twin_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        if SET_YLIM:
            twin_ax.set_ylim(0, 500)

    if PLOT_POWER_CURVE:
        ax.plot(observed_flows, observed_powers, marker='o', color='b', linestyle='-')
    if num_dams_plot > 1:
        dam_name = DAM_NAMES[dam_id] if dam_id in DAM_NAMES else dam_id
        ax.set_title(f'Power Curve function of the {dam_name}{plot_title}')
    ax.set_xlabel('Turbine Flow (m$^3$/s)')
    ax.set_ylabel('Power (MW)')
    ax.grid(True)

# Trabs
plt.tight_layout()
if not USE_TRANSPARENCY:
    plt.savefig(filename + '.eps')
plt.savefig(filename + '.png')
plt.show()

