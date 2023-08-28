from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from datetime import datetime, time

path_daily_inflow_data = "../data/history/historical_data_daily_avg_inflow.pickle"
daily_inflow_data = pd.read_pickle(path_daily_inflow_data)
print(daily_inflow_data)

# Small exploratory analysis
print(daily_inflow_data['total_avg_inflow'].min())
print(daily_inflow_data['total_avg_inflow'].mean())
print(daily_inflow_data['total_avg_inflow'].max())

# Average inflow of the instances I am familiar with
print("Instance 1:", daily_inflow_data.loc[datetime.strptime("2021-04-03", "%Y-%m-%d").date(), 'total_avg_inflow'])
print("Instance 3:", daily_inflow_data.loc[datetime.strptime("2020-12-01", "%Y-%m-%d").date(), 'total_avg_inflow'])

# Sort by total avg inflow
sorted_daily_inflow = daily_inflow_data.sort_values(by='total_avg_inflow')
print(sorted_daily_inflow)

# Select percentiles 0% 10% .. 100%
percentile_indices = [int((percentile / 100) * (len(sorted_daily_inflow) - 1)) for percentile in range(0, 101, 10)]
print(percentile_indices)
selected_rows = sorted_daily_inflow.iloc[percentile_indices]
print(selected_rows)
selected_dates = selected_rows.index.tolist()
print(selected_dates)

# Create the instances
path_constants = f"../data/constants/constants_2dams.json"
path_historical_data = "../data/history/historical_data.pickle"
for index, date in enumerate(selected_dates):
    instance = RLEnvironment.create_instance(
        length_episodes=24 * 4 + 3,  # One day (+ impact buffer)
        constants=load_json(path_constants),
        historical_data=pd.read_pickle(path_historical_data),
        initial_row=datetime.combine(date, time(0, 0)),
    )
    inconsistencies = instance.check()
    if inconsistencies:
        raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

    instance.to_json(f"instances_base/instance_intermediate{index}.json")
