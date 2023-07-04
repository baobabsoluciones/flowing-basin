from flowing_basin.core import Instance, Solution
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from datetime import timedelta

INSTANCE = 1
Y_VALUES = "out_flows"
SOLUTION = "MILP"
NUM_DAMS = 8
NO_TEXT = True

instance = Instance.from_json(f"../data/input_example{INSTANCE}_{NUM_DAMS}dams_1days.json")
if SOLUTION == "MILP":
    solution = Solution.from_json(f"../data/output_instance{INSTANCE}_LPmodel_V2_{NUM_DAMS}dams_1days.json")
elif SOLUTION == "PSO_k=2":
    if INSTANCE == 1:
        if NUM_DAMS == 2:
            solution = Solution.from_json(
                f"../data/output_instance1_PSO_2dams_1days_2023-05-17 21.36_mode=linear_k=2/solution.json"
            )
        elif NUM_DAMS == 3:
            solution = Solution.from_json(
                f"../data/output_instance1_PSO_3dams_1days_2023-05-17 21.55_mode=linear_k=2/solution.json"
            )
        elif NUM_DAMS == 8:
            solution = Solution.from_json(
                f"../data/output_instance1_PSO_8dams_1days_2023-05-18 00.12_mode=linear_k=2/solution.json"
            )
        else:
            raise FileNotFoundError()
    elif INSTANCE == 3:
        if NUM_DAMS == 2:
            solution = Solution.from_json(
                f"../data/output_instance3_PSO_2dams_1days_2023-05-17 17.56_mode=linear_k=2/solution.json"
            )
        else:
            raise FileNotFoundError()
    else:
        raise FileNotFoundError()
elif SOLUTION == "PSO_k=0":
    if INSTANCE == 1:
        if NUM_DAMS == 2:
            solution = Solution.from_json(
                f"../data/output_instance1_PSO_2dams_1days_2023-07-03 16.03_mode=linear_k=0/solution.json"
            )
        else:
            raise FileNotFoundError()
    else:
        raise FileNotFoundError()
else:
    raise ValueError()

# Get X axis, time ---- #

start_date, end_date = instance.get_start_end_datetimes()
print(start_date, end_date)
times = []
current_date = start_date
while current_date <= end_date:
    times.append(current_date)
    current_date += timedelta(minutes=15)

times = [time.strftime("%Y-%m-%d %H:%M") for time in times]
print(f"{times=}")

times = mdates.datestr2num(times)
print(f"{times=}")
num_times = len(times)

# Get Y axis ---- #

# Prices
prices = instance.get_all_prices()
prices = prices[0:num_times]
print(f"{prices=}")

# Incoming flows
in_flows = dict()
for dam_id in instance.get_ids_of_dams():
    if instance.get_order_of_dam(dam_id) == 1:
        in_flows[dam_id] = [incoming + unregulated for incoming, unregulated in zip(
            instance.get_all_unregulated_flows_of_dam(dam_id),
            instance.get_all_incoming_flows()
        )]
    else:
        in_flows[dam_id] = instance.get_all_unregulated_flows_of_dam(dam_id)
    in_flows[dam_id] = in_flows[dam_id][0:num_times]
print(f"{in_flows=}")

# > Average incoming flows through all dams (used when NUM_DAMS > 2)
avg_in_flows = [
    sum(in_flows[dam_id][i] for dam_id in instance.get_ids_of_dams()) / instance.get_num_dams()
    for i in range(num_times)
]
print(f"{avg_in_flows=}")

# Exiting flows
out_flows = dict()
for dam_id in instance.get_ids_of_dams():
    out_flows[dam_id] = solution.get_exiting_flows(dam_id)
    out_flows[dam_id] = out_flows[dam_id][0:num_times]
print(f"{out_flows=}")

# > Average exiting flows through all dams (used when NUM_DAMS > 2)
avg_out_flows = [
    sum(out_flows[dam_id][i] for dam_id in instance.get_ids_of_dams()) / instance.get_num_dams()
    for i in range(num_times)
]
print(f"{avg_out_flows=}")

# Plot ---- #

fig, ax = plt.subplots(1, 1)
twin_ax = None
if Y_VALUES == "out_flows":
    twin_ax = ax.twinx()  # to plot prices alongside out flows

# Labels
if not NO_TEXT:
    # Y label
    if Y_VALUES == "prices":
        ax.set_ylabel("Price (€/MWh)", fontsize=12)
    elif Y_VALUES == "in_flows":
        if NUM_DAMS < 3:
            ax.set_ylabel("Incoming flow (m3/s)", fontsize=12)
        else:
            ax.set_ylabel("Average incoming flow (m3/s)", fontsize=12)
    elif Y_VALUES == "out_flows":
        if NUM_DAMS < 3:
            ax.set_ylabel("Exiting flow (m3/s)", fontsize=12)
        else:
            ax.set_ylabel("Average exiting flow (m3/s)", fontsize=12)
        twin_ax.set_ylabel("Price (€/MWh)")
    else:
        raise ValueError()
    # X label
    ax.set_xlabel("Time", fontsize=12)

# X ticks
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.tick_params(axis='x', labelsize=10)

# Y ticks, grid
ax.tick_params(axis='y', labelsize=10)
ax.grid('on')

# Plot
if Y_VALUES == "prices":
    ax.plot(times, prices, linewidth=2, color="red")
elif Y_VALUES == "in_flows":
    if NUM_DAMS < 3:
        for dam_id in instance.get_ids_of_dams():
            ax.plot(times, in_flows[dam_id], linewidth=2, color="blue")
            if not NO_TEXT:
                ax.text(
                    times[-16], in_flows[dam_id][-1] + 0.75,
                    f"Dam {instance.get_order_of_dam(dam_id)}", fontsize=14, color="blue", va="center"
                )
    else:
        ax.plot(times, avg_in_flows, linewidth=2, color="blue")
    ax.set_ylim(0, 17.5)
elif Y_VALUES == "out_flows":
    if NUM_DAMS < 3:
        for dam_id in instance.get_ids_of_dams():
            ax.plot(times, out_flows[dam_id], linewidth=2, color="green",
                    linestyle="dashed" if dam_id == "dam2" else "solid")
            if not NO_TEXT:
                ax.text(
                    times[-16], out_flows[dam_id][-1] + 0.75,
                    f"Dam {instance.get_order_of_dam(dam_id)}", fontsize=14, color="green", va="center"
                )
    else:
        ax.plot(times, avg_out_flows, linewidth=2, color="green")
    twin_ax.plot(times, prices, linewidth=1.33, color="red")
else:
    raise ValueError()

plt.savefig(f"instance_plots/instance{INSTANCE}/{str(NUM_DAMS) + 'dams_' if NUM_DAMS > 2 else ''}"
            f"{Y_VALUES}{'_' + SOLUTION if Y_VALUES == 'out_flows' else ''}{'_no_text' if NO_TEXT else ''}.png")
plt.show()
