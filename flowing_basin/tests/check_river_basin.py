from test_river_basin import check_income, check_volume
from flowing_basin.tools import RiverBasin
from flowing_basin.solvers.rl import Training
from cornflow_client.core.tools import load_json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os

LENGTH_EPISODE = 24 * 4
START = datetime.strptime("2021-09-29 01:00", "%Y-%m-%d %H:%M")
# START = None
# Use None if you want a random starting datetime
CHECK_INCOMES = True
CHECK_VOLUMES = True

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

vols_calc = {
    dam_id: [] for dam_id in instance.get_ids_of_dams()
}
turbined_flows_calc = {
    dam_id: [] for dam_id in instance.get_ids_of_dams()
}
powers_calc = {
    dam_id: [] for dam_id in instance.get_ids_of_dams()
}

initial_state = river_basin.get_state()
old_state = initial_state

for dam_id in instance.get_ids_of_dams():
    vols_calc[dam_id].append(initial_state[dam_id]["vol"])
    turbined_flows_calc[dam_id].append(initial_state[dam_id]["turbined_flow"])
    powers_calc[dam_id].append(initial_state[dam_id]["power"])

decisions = [
    [dam1_flow, dam2_flow]
    for dam1_flow, dam2_flow in zip(df["dam1_flow"].loc[initial_row: last_row],
                                    df["dam2_flow"].loc[initial_row: last_row])
]
print(decisions)

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

    for dam_id in instance.get_ids_of_dams():
        vols_calc[dam_id].append(state[dam_id]["vol"])
        turbined_flows_calc[dam_id].append(state[dam_id]["turbined_flow"])
        powers_calc[dam_id].append(state[dam_id]["power"])

# Join the calculated volumes, turbined flows and powers with the real values in a data frame ---- #

df1 = pd.DataFrame()

df1["datetime"] = np.array(df["datetime"].loc[initial_row - 1: last_row])

for dam_id in instance.get_ids_of_dams():

    df1[dam_id + "_flow"] = np.array(df[dam_id + "_flow"].loc[initial_row - 1: last_row])

    # The volume stored in data is the initial volume of every period (and not the final volume, as we do)
    # This is why we shift the column of the data by one row, so it represents the final volume in every time step
    df1[dam_id + "_vol_REAL"] = np.array(df[dam_id + "_vol"].loc[initial_row: last_row + 1])
    df1[dam_id + "_vol_CALC"] = np.array(vols_calc[dam_id])

    df1[dam_id + "_turbined_flow_REAL"] = np.array(df[dam_id + "_turbined_flow"].loc[initial_row - 1: last_row])
    df1[dam_id + "_turbined_flow_CALC"] = np.array(turbined_flows_calc[dam_id])

    df1[dam_id + "_power_REAL"] = np.array(df[dam_id + "_power"].loc[initial_row - 1: last_row])
    df1[dam_id + "_power_CALC"] = np.array(powers_calc[dam_id])

    df1[dam_id + "_unreg_flow"] = np.array(df[dam_id + "_unreg_flow"].loc[initial_row - 1: last_row])

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
path = f"river_basin_checks/check_{start.strftime('%Y-%m-%d_%H-%M')}.png"
while os.path.exists(path):
    version += 1
    path = f"river_basin_checks/check_{start.strftime('%Y-%m-%d_%H-%M')}_v{version}.png"
print(path)
fig.savefig(fname=path)
plt.close()
