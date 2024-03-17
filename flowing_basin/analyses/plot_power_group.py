from cornflow_client.core.tools import load_json
from flowing_basin.tools import PowerGroup
from flowing_basin.core import Instance
from flowing_basin.solvers.rl import ReinforcementLearning
import matplotlib.pyplot as plt
import numpy as np


def lighten_color(color, amount=0.5):
    """
    Function from StackOverflow (https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib)
    Returns a lighter (amount<1) or darker (amount>1) version of the color
    Examples:
    >> lighten_color('green', 0.3)
    # Returns a color that is like the 'green' color, but lighter
    >> lighten_color('green', 1.3)
    # Returns a color that is like the 'green' color, but darker
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


PLOT_MILP_FLOWS = True
if PLOT_MILP_FLOWS:
    GENERAL = 'G1'
    filename_milp = '_vs_milp_flows'
    plot_title = f' with MILP flows histogram in {GENERAL}'
else:
    filename_milp = ''
    plot_title = ''


filename = f'power_group_charts/power_vs_turbine_flow{filename_milp}.eps'
constants = Instance.from_dict(load_json(ReinforcementLearning.constants_path))

if PLOT_MILP_FLOWS:
    milp_flows = {dam_id: [] for dam_id in constants.get_ids_of_dams()}
    for sol in ReinforcementLearning.get_all_baselines(GENERAL):
        if sol.get_solver() == "MILP":
            for dam_id in sol.get_ids_of_dams():
                milp_flows[dam_id].extend(sol.get_exiting_flows_of_dam(dam_id))
else:
    milp_flows = None

fig, axs = plt.subplots(1, constants.get_num_dams(), figsize=(12, 5))
for i, dam_id in enumerate(constants.get_ids_of_dams()):

    data = constants.get_turbined_flow_obs_for_power_group(dam_id)
    observed_flows = data['observed_flows']
    observed_powers = data['observed_powers']
    max_flow = max(observed_flows)

    startup_flows = constants.get_startup_flows_of_power_group(dam_id)
    shutdown_flows = constants.get_shutdown_flows_of_power_group(dam_id)

    flow_bins = PowerGroup.get_turbined_bins_and_groups(startup_flows, shutdown_flows)
    print(dam_id, "flow bins:", flow_bins)

    ax = axs[i]
    ax.plot(observed_flows, observed_powers, marker='o', color='b', linestyle='-')
    ax.set_title(f'Power group dynamics of {dam_id}{plot_title}')
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

    if PLOT_MILP_FLOWS:
        flows_limits = flow_bins[0]
        flows_limits = np.append(0, np.append(flows_limits, max_flow))
        bins = []
        for j in range(len(flows_limits) - 1):
            bins.extend(np.linspace(flows_limits[j], flows_limits[j + 1], 6, endpoint=False))
        bins.append(flows_limits[-1])
        print(dam_id, "bins:", bins)
        twin_ax = ax.twinx()
        twin_ax.hist(milp_flows[dam_id], color='orange', alpha=0.5, bins=bins)
        twin_ax.set_ylabel('MILP flows frequency')

# Trabs
plt.tight_layout()
plt.savefig(filename, format='eps')
plt.show()

