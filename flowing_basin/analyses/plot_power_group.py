"""
plot_power_group.py
This script draws the power curve of both dams,
optionally adding a histogram of the outflows of a solver
"""

from cornflow_client.core.tools import load_json
from flowing_basin.tools import PowerGroup
from flowing_basin.core import Instance
from flowing_basin.core.utils import lighten_color
from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.solvers.common import CONSTANTS_PATH
import matplotlib.pyplot as plt
import numpy as np


PLOT_SOLVER_FLOWS = True
DAM_IDS = None  # Put None to plot all dams, or [dam_id] for a single dam
SOLVER = "MILP"  # an agent, e.g. "rl-A113G1O232R22T3", or "MILP"
GENERAL = 'G0'  # only matters if PLOT_SOLVER_FLOWS = True and SOLVER = "MILP"
FILENAME = f'plot_power_group/fig_power_vs_turbine_flow'
DAM_NAMES = {'dam1': 'first subsystem', 'dam2': 'second subsystem'}

if PLOT_SOLVER_FLOWS:
    filename_solver = f'_vs_{SOLVER}_outflows'
    plot_title = f' with {SOLVER} outflows'
    # if SOLVER == "MILP":
    #     plot_title += f' in {GENERAL}'
else:
    filename_solver = ''
    plot_title = ''

constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=2)))
if DAM_IDS is None:
    DAM_IDS = constants.get_ids_of_dams()
    filename_dams = ''
else:
    assert isinstance(DAM_IDS, list)
    filename_dams = '_' + '_'.join(DAM_IDS)
filename = FILENAME + filename_solver + filename_dams

if PLOT_SOLVER_FLOWS and SOLVER == "MILP":
    solver_flows = {dam_id: [] for dam_id in constants.get_ids_of_dams()}
    for sol in ReinforcementLearning.get_all_baselines(GENERAL):
        if sol.get_solver() == "MILP":
            for dam_id in sol.get_ids_of_dams():
                solver_flows[dam_id].extend(sol.get_exiting_flows_of_dam(dam_id))
elif PLOT_SOLVER_FLOWS:
    solver_flows = {dam_id: [] for dam_id in constants.get_ids_of_dams()}
    rl = ReinforcementLearning(SOLVER)
    runs = rl.run_agent(ReinforcementLearning.get_all_fixed_instances(num_dams=rl.config.num_dams))
    for run in runs:
        sol = run.solution
        for dam_id in sol.get_ids_of_dams():
            solver_flows[dam_id].extend(sol.get_exiting_flows_of_dam(dam_id))
else:
    solver_flows = None

fig, axs = plt.subplots(1, len(DAM_IDS), figsize=(6 * len(DAM_IDS), 5))
for i, dam_id in enumerate(DAM_IDS):

    data = constants.get_turbined_flow_obs_for_power_group(dam_id)
    observed_flows = data['observed_flows']
    observed_powers = data['observed_powers']
    max_flow = max(observed_flows)

    startup_flows = constants.get_startup_flows_of_power_group(dam_id)
    shutdown_flows = constants.get_shutdown_flows_of_power_group(dam_id)

    flow_bins = PowerGroup.get_turbined_bins_and_groups(startup_flows, shutdown_flows)
    print(dam_id, "flow bins:", flow_bins)

    ax = axs[i] if len(DAM_IDS) > 1 else axs
    ax.plot(observed_flows, observed_powers, marker='o', color='b', linestyle='-')
    if len(DAM_IDS) > 1:
        ax.set_title(f'Power group dynamics of {DAM_NAMES[dam_id]}{plot_title}')
    ax.set_xlabel('Turbine flow (m3/s)')
    ax.set_ylabel('Power (MW)')
    ax.grid(True)

    flows, groups = flow_bins
    i = 0
    while i < len(flows):

        # Shaded area
        limits = (flows[i], flows[i + 1]) if i < len(flows) - 1 else (flows[i], max_flow)
        x = np.linspace(limits[0], limits[1])
        y = np.interp(x=x, xp=observed_flows, fp=observed_powers)
        col = 'lightgreen' if i % 2 == 0 else 'lightcoral'
        ax.fill_between(x, y, facecolor=lighten_color(col))

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
        twin_ax.hist(solver_flows[dam_id], color='orange', alpha=0.5, bins=bins, label=f"{SOLVER} outflows")
        twin_ax.set_ylabel(f'{SOLVER} outflows frequency')
        twin_ax.legend()

# Trabs
plt.tight_layout()
plt.savefig(filename + '.eps')
plt.savefig(filename + '.png')
plt.show()

