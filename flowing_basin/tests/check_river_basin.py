from test_river_basin import check_income, check_volume
from flowing_basin.tools import RiverBasin
from flowing_basin.solvers.rl import Training
from cornflow_client.core.tools import load_json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os


# Option to check consistency throughout states after every update of the river basin
CHECK_INCOMES = True
CHECK_VOLUMES = True

# Path and extension in which to save the generated graphs
PATH_START = "river_basin_checks/check_"
PATH_END = ""
EXTENSION = ".png"

# Episode length and starting date
LENGTH_EPISODE = 24 * 4  # One day
START = datetime.strptime("2020-12-01 05:45", "%Y-%m-%d %H:%M")
# START = None
# Use None if you want a random starting datetime

# Number of log lines (of the river basin and of the data) that should be printed
# LOG_LINES = 11
LOG_LINES = LENGTH_EPISODE

# Observations:
# 2021-09-29 01:00
# - The power models return a power greater than 0 (0.49487926 and 1.0269122) even for null past flows
# - In this episode, the real volume of dam1 increases at the end,
#   when this shouldn't happen because flow < incoming + unreg (there may because of variables ignored by our model)
# - In this episode, the volume of dam2 is clipped at its maximum value, 58343,
#   which is why the volume of our model looks like nothing like the real volume
#   >> Actually, the volume of our model is very different to the data, even if it isn't clipped;
#   the real reason is an inconsistency with the data volume and the flows that enter and exit the dam
# 2020-04-09 03:15
# - A clear example of how the model greatly diverges from the real data when volumes are clipped
#   >> Again, the divergence also happens when volumes aren't clipped; the real reason of the divergence is that
#   with the low incoming and unregulated flows, and high exiting flows, the volume should be dropping like in our model
# 2021-08-31 10:30
# - Like in 2021-09-29 01:00, in this episode the real volume of dam2 keeps steady,
#   when it should actually be increasing with the unregulated and turbined flow it gets (as seen in our model)
#   Something similar (although less striking) happens with the volume of dam1

# ---- CODE ---- #

# Get data and create environment - we will compare the results of the environment with the data ---- #

# Create instance
path_constants = "../data/rl_training_data/constants.json"
path_training_data = "../data/rl_training_data/training_data.pickle"
df = pd.read_pickle(path_training_data)
instance = Training.create_instance(
    length_episodes=LENGTH_EPISODE,
    constants=load_json(path_constants),
    training_data=df,
    initial_row=START,
)
print(instance.data)

# Get necessary values from instance
start, _ = instance.get_start_end_datetimes()
initial_row = df.index[df["datetime"] == start].tolist()[0]
last_row = initial_row + LENGTH_EPISODE - 1
max_last_lag = max(
    instance.get_relevant_lags_of_dam(dam_id)[-1]
    for dam_id in instance.get_ids_of_dams()
)
print(start)

# Create river basin
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)

# Calculate volumes, turbined flows and powers with the decisions saved in data ---- #

# Variables to record - volumes, turbined flows, powers and incomes
vols_calc = {dam_id: [] for dam_id in instance.get_ids_of_dams()}
turbined_flows_calc = {dam_id: [] for dam_id in instance.get_ids_of_dams()}
powers_calc = {dam_id: [] for dam_id in instance.get_ids_of_dams()}
incomes = []

# Append initial values
initial_state = river_basin.get_state()
old_state = initial_state
for dam_id in instance.get_ids_of_dams():
    vols_calc[dam_id].append(initial_state[dam_id]["vol"])
    turbined_flows_calc[dam_id].append(initial_state[dam_id]["turbined_flow"])
    powers_calc[dam_id].append(initial_state[dam_id]["power"])
incomes.append(None)

# Decisions taken in data
decisions = [
    [dam1_flow, dam2_flow]
    for dam1_flow, dam2_flow in zip(
        df["dam1_flow"].loc[initial_row:last_row],
        df["dam2_flow"].loc[initial_row:last_row],
    )
]
print(decisions)

# Create log of real values
log_real = river_basin.create_log()

for row, flows in enumerate(decisions):

    income = river_basin.update(flows)
    state = river_basin.get_state()
    if CHECK_VOLUMES:
        check_volume(
            old_state=old_state,
            state=state,
            time_step=river_basin.instance.get_time_step(),
            min_vol_dam1=river_basin.instance.get_min_vol_of_dam("dam1"),
            max_vol_dam1=river_basin.instance.get_max_vol_of_dam("dam1"),
            min_vol_dam2=river_basin.instance.get_min_vol_of_dam("dam2"),
            max_vol_dam2=river_basin.instance.get_max_vol_of_dam("dam2"),
        )
    if CHECK_INCOMES:
        check_income(
            old_state=old_state,
            state=state,
            time_step=river_basin.instance.get_time_step(),
            income=income,
        )
    old_state = state

    # Add values of new state to lists
    for dam_id in instance.get_ids_of_dams():
        vols_calc[dam_id].append(state[dam_id]["vol"])
        turbined_flows_calc[dam_id].append(state[dam_id]["turbined_flow"])
        powers_calc[dam_id].append(state[dam_id]["power"])
    incomes.append(income)

    # Add log of real values
    incoming_flow = df['incoming_flow'].loc[initial_row + row]
    log_real += f"\n{row: ^6}{round(incoming_flow, 2): ^13}"
    turbined_flow_of_preceding_dam = 0  # Any value (it will not be used for dam1)
    power_total = 0
    for dam_index, dam_id in enumerate(instance.get_ids_of_dams()):
        flow_out = df[dam_id + '_flow'].loc[initial_row + row]
        unregulated_flow = df[dam_id + '_unreg_flow'].loc[initial_row + row]
        turbined_flow = df[dam_id + '_turbined_flow'].loc[initial_row + row]
        volume = df[dam_id + '_vol'].loc[initial_row + row + 1]  # Remember that data's volumes are offset by one row
        power = df[dam_id + '_power'].loc[initial_row + row]
        power_total += power
        net_flow = (
            incoming_flow + unregulated_flow - flow_out if dam_index == 0 else
            turbined_flow_of_preceding_dam + unregulated_flow - flow_out
        )
        log_real += (
            f"{round(unregulated_flow, 4): ^13}{round(flow_out, 4): ^14}"
            f"{round(flow_out, 4): ^14}{round(flow_out, 4): ^14}"
            f"{round(net_flow, 4): ^13}{round(net_flow * instance.get_time_step(), 5): ^15}"
            f"{round(volume, 2): ^13}{round(power, 2): ^13}"
            f"|\t"
            f"{round(turbined_flow, 5): ^15}"
        )
        turbined_flow_of_preceding_dam = turbined_flow
    price = df['price'].loc[initial_row + row]
    income_real = price * power_total * instance.get_time_step() / 3600
    log_real += f"{round(price, 2): ^13}{round(income_real, 2): ^13}"

# Print river basin log and data (real) logs
print("--- river basin log")
print("note: time 0 corresponds to the row 1 of the data frames")
print("\n".join(river_basin.log.split("\n")[0: LOG_LINES + 1]))
print("--- data log")
print("note: time 0 corresponds to the row 1 of the data frames")
print("\n".join(log_real.split("\n")[0: LOG_LINES + 1]))

# Join the calculated volumes, turbined flows and powers with the real values in a data frame ---- #

df1 = pd.DataFrame()

df1["datetime"] = np.array(df["datetime"].loc[initial_row - 1 : last_row])
df1["incoming_flow"] = np.array(
    df["incoming_flow"].loc[initial_row - 1: last_row]
)

for dam_id in instance.get_ids_of_dams():

    df1[dam_id + "_unreg_flow"] = np.array(
        df[dam_id + "_unreg_flow"].loc[initial_row - 1: last_row]
    )

    df1[dam_id + "_flow"] = np.array(
        df[dam_id + "_flow"].loc[initial_row - 1: last_row]
    )

    # The volume stored in data is the initial volume of every period (and not the final volume, as we do)
    # This is why we shift the column of the data by one row, so it represents the final volume in every time step
    df1[dam_id + "_vol_REAL"] = np.array(
        df[dam_id + "_vol"].loc[initial_row : last_row + 1]
    )
    df1[dam_id + "_vol_CALC"] = np.array(vols_calc[dam_id])

    df1[dam_id + "_turbined_flow_REAL"] = np.array(
        df[dam_id + "_turbined_flow"].loc[initial_row - 1 : last_row]
    )
    df1[dam_id + "_turbined_flow_CALC"] = np.array(turbined_flows_calc[dam_id])

    df1[dam_id + "_power_REAL"] = np.array(
        df[dam_id + "_power"].loc[initial_row - 1 : last_row]
    )
    df1[dam_id + "_power_CALC"] = np.array(powers_calc[dam_id])

df1["price"] = np.array(df["price"].loc[initial_row - 1 : last_row])
df1["income_REAL"] = np.array(
    (df1["dam1_power_REAL"] + df1["dam2_power_REAL"])
    * instance.get_time_step()
    / 3600
    * df1["price"]
)
df1["income_CALC"] = np.array(incomes)

print(df1)

# Plot difference between real and calculated values ---- #

# Create and show plot
fig, axs = plt.subplots(2, 3)
for i, dam in enumerate(["dam1", "dam2"]):
    for j, concept in enumerate(["vol", "turbined_flow", "power"]):
        df1[[dam + "_" + concept + "_REAL", dam + "_" + concept + "_CALC"]].plot(
            ax=axs[i, j]
        )
plt.show()

# Save plot
version = 0
path = PATH_START + str(start.strftime('%Y-%m-%d_%H-%M')) + PATH_END + EXTENSION
while os.path.exists(path):
    version += 1
    path = PATH_START + str(start.strftime('%Y-%m-%d_%H-%M'))+ PATH_END + f"_v{version}" + EXTENSION
print(path)
fig.savefig(fname=path)
plt.close()
