from cornflow_client.core.tools import load_json
from flowing_basin.solvers.rl import RLEnvironment
import pandas as pd
from datetime import datetime, time

CREATE_INSTANCES = True
PERCENTILES = list(range(0, 101, 10))  # percentiles 0% 10% .. 100% from driest to rainiest

path_daily_inflow_data = "../data/history/historical_data_daily_avg_inflow.pickle"
daily_inflow_data = pd.read_pickle(path_daily_inflow_data)
# print(daily_inflow_data)

# Small exploratory analysis
print("Min avg inflow:", daily_inflow_data['total_avg_inflow'].min())
print("Mean avg inflow:", daily_inflow_data['total_avg_inflow'].mean())
print("Max avg inflow:", daily_inflow_data['total_avg_inflow'].max())

# Average inflow of the instances I am familiar with, instance1 and instance3
print("Instance 1 avg inflow:", daily_inflow_data.loc[datetime.strptime("2021-04-03", "%Y-%m-%d").date(), 'total_avg_inflow'])
print("Instance 3 avg inflow:", daily_inflow_data.loc[datetime.strptime("2020-12-01", "%Y-%m-%d").date(), 'total_avg_inflow'])

# Sort by total avg inflow (in ASCENDING order, so first dates are very dry)
sorted_daily_inflow = daily_inflow_data.sort_values(by='total_avg_inflow')
# print(sorted_daily_inflow)

# Select the indices of the given percentiles (from driest to rainiest)
percentile_indices = [int((percentile / 100) * (len(sorted_daily_inflow) - 1)) for percentile in PERCENTILES]
print("Percenile indeces:", percentile_indices)
selected_rows = sorted_daily_inflow.iloc[percentile_indices]
# print("Selected rows:", selected_rows)
selected_dates = selected_rows.index.tolist()
print("Selected dates:", selected_dates)
print("Corresponding avg inflows:", [sorted_daily_inflow.loc[date, 'total_avg_inflow'] for date in selected_dates])

# Percentile position of instance1 and instance3
print("Instance 1 percentile:", sorted_daily_inflow.index.get_loc(datetime.strptime("2021-04-03", "%Y-%m-%d").date()) / len(daily_inflow_data.index))
print("Instance 3 percentile:", sorted_daily_inflow.index.get_loc(datetime.strptime("2020-12-01", "%Y-%m-%d").date()) / len(daily_inflow_data.index))

# Create the instances for percentiles 0% 10% .. 100%
if CREATE_INSTANCES:
    path_constants = f"../data/constants/constants_2dams.json"
    path_historical_data = "../data/history/historical_data.pickle"
    for index, date in enumerate(selected_dates):
        instance_path = f"instances_base/instancePercentile{PERCENTILES[index]:>02}.json"
        instance = RLEnvironment.create_instance(
            length_episodes=24 * 4 + 3,  # One day (+ impact buffer)
            constants=load_json(path_constants),
            historical_data=pd.read_pickle(path_historical_data),
            initial_row=datetime.combine(date, time(0, 0)),
        )
        inconsistencies = instance.check()
        if inconsistencies:
            raise Exception(f"There are inconsistencies in {instance_path}: {inconsistencies}")
        instance.to_json(instance_path)
        print(f"Created {instance_path}.")
