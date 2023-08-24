from flowing_basin.tools import RiverBasin
from flowing_basin.solvers.rl import RLEnvironment
from cornflow_client.core.tools import load_json
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os

# Path and extension in which to save the generated graphs
SAVE_PLOT = False
PATH_START = "simulation_vs_history_graphs/sim_v_hist_NO-NA_"
PATH_END = ""
EXTENSION = ".png"

# Episode length and starting date
LENGTH_EPISODE = 24 * 4  # One day
START = datetime.strptime("2021-09-29 01:00", "%Y-%m-%d %H:%M")
# START = None
# Use None if you want a random starting datetime

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
# NOTE: These problems arise both with historical_data.pickle
# and with historical_data_reliable_only.pickle (in which rows with unknown unregulated flows were eliminated and not set to 0)
# Although the inconsistency problem is less often (actually, very uncommon) with the second dataset

# ---- CODE ---- #

# Variable values in HISTORY ---- #

# Create instance
path_constants = "../data/rl_training_data/constants_2dams.json"
path_training_data = "../data/rl_training_data/historical_data_reliable_only.pickle"
df = pd.read_pickle(path_training_data)
instance = RLEnvironment.create_instance(
    length_episodes=LENGTH_EPISODE,
    constants=load_json(path_constants),
    historical_data=df,
    initial_row=START,
)
print(instance.data)

# Get necessary values from instance
start, _ = instance.get_start_end_datetimes()
initial_row = df.index[df["datetime"] == start].tolist()[0]
last_row = initial_row + LENGTH_EPISODE - 1
print(start)

# Decisions taken in data
decisions = np.array(
    [
        [[dam1_flow], [dam2_flow]]
        for dam1_flow, dam2_flow in zip(
            df["dam1_flow"].loc[initial_row:last_row],
            df["dam2_flow"].loc[initial_row:last_row],
        )
    ]
)
print(decisions)

# Values of volume, unregulated flow, etc.
df_history = df.loc[initial_row:last_row]
df_history.reset_index(drop=True, inplace=True)
print(df_history)

# Variable values calculated by our SIMULATION MODEL ---- #

# Create river basin
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)

# Run simulation model
river_basin.deep_update(decisions, is_relvars=False, fast_mode=False)

# Values of volume, unregulated flow, etc.
df_simulation = river_basin.history
for dam_id in instance.get_ids_of_dams():
    df_simulation.columns = df_simulation.columns.str.replace(f'{dam_id}_flow_clipped2', f'{dam_id}_flow')
print(df_history)

# Compare HISTORY with SIMULATION MODEL ---- #

# Print river basin log and data (real) logs
print("--- river basin")
print(df_simulation.to_string())
print("--- data log")
print(df_history.to_string())

# Plot difference between real and calculated values

# Create plot - code taken from RiverBasin's plot_history method
fig, axs = plt.subplots(2, 3)
fig.set_size_inches(10 * 3, 10)
for i in range(2):
    for j in range(3):
        axs[i, j].set_xlabel("Time (15min)")
    axs[i, 0].set_ylabel("Volume (m3)")
    axs[i, 1].set_ylabel("Flow (m3/s)")
    axs[i, 2].set_ylabel("Power (MW)")
for i, dam_id in enumerate(instance.get_ids_of_dams()):
    for j, var in enumerate(["vol", "flow", "power"]):
        axs[i, j].plot(df_history[f"{dam_id}_{var}"], label=f"{dam_id}_{var}")
        axs[i, j].plot(df_simulation[f"{dam_id}_{var}"], label=f"{dam_id}_{var}_sim")
decision_horizon = instance.get_decision_horizon()
for i in range(2):
    for j in range(3):
        axs[i, j].axvline(x=decision_horizon, color='gray')
        axs[i, j].legend()

# Save plot
if SAVE_PLOT:
    version = 0
    path = PATH_START + str(start.strftime('%Y-%m-%d_%H-%M')) + PATH_END + EXTENSION
    while os.path.exists(path):
        version += 1
        path = PATH_START + str(start.strftime('%Y-%m-%d_%H-%M')) + PATH_END + f"_v{version}" + EXTENSION
    print(path)
    plt.savefig(fname=path)
    plt.close()

# Show plot
plt.show()
