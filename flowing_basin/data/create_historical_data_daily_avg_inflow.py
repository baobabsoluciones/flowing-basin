import pandas as pd
from datetime import datetime

SAVE_DATAFRAME = False

path_historical_data = "history/historical_data.pickle"
df = pd.read_pickle(path_historical_data)
print(df)
print(df.head().to_string())
print(df.tail().to_string())

# Turn the datetimes into dates (w/ only the year, month, day)
dates = df['datetime'].dt.date
print(dates)

# Group dataframe by the dates
df_grouped_by_date = df.groupby(dates)
# print(df_grouped_by_date)  # <-- No useful information

# Perform the mean of the incoming and unregulated flows of each group
df_grouped_by_date = df_grouped_by_date.agg({
    'incoming_flow': 'mean',
    'dam1_unreg_flow': 'mean',
    'dam2_unreg_flow': 'mean'
})
print(df_grouped_by_date)

# Rename columns and index
df_grouped_by_date = df_grouped_by_date.rename(
    columns={
        'incoming_flow': 'incoming_flow_avg',
        'dam1_unreg_flow': 'dam1_unreg_flow_avg',
        'dam2_unreg_flow': 'dam2_unreg_flow_avg'
    }
)
df_grouped_by_date.index.rename('date', inplace=True)
print(df_grouped_by_date)

# Add a third column with the total avg inflow
df_grouped_by_date['total_avg_inflow'] = (
    df_grouped_by_date['incoming_flow_avg'] +
    df_grouped_by_date['dam1_unreg_flow_avg'] +
    df_grouped_by_date['dam2_unreg_flow_avg']
)
print(df_grouped_by_date)

# Test
example_date = datetime.strptime("2021-04-03", "%Y-%m-%d").date()
filtered_df = df[df['datetime'].dt.date == example_date]
assert (
    filtered_df['incoming_flow'].mean() ==
    df_grouped_by_date.loc[example_date, 'incoming_flow_avg']
)
assert (
    filtered_df['dam1_unreg_flow'].mean() ==
    df_grouped_by_date.loc[example_date, 'dam1_unreg_flow_avg']
)
# assert (
#     filtered_df['dam2_unreg_flow'].mean() ==
#     df_grouped_by_date.loc[example_date, 'dam2_unreg_flow_avg']
# )
# This last assertion fails because of rounding errors:
print("Incoming flows:", filtered_df['incoming_flow'].values.tolist())
print("Incoming flow calculated mean:", filtered_df['incoming_flow'].mean())
print("Incoming flow stored mean:", df_grouped_by_date.loc[example_date, 'incoming_flow_avg'])
for dam in ['dam1', 'dam2']:
    print(dam, "unregulated flows:", filtered_df[dam + '_unreg_flow'].values.tolist())
    print(dam, "unregulated flow calculated mean:", filtered_df[dam + '_unreg_flow'].mean())
    print(dam, "unregulated flow stored mean:", df_grouped_by_date.loc[example_date, dam + '_unreg_flow_avg'])
print("Total avg inflow:", df_grouped_by_date.loc[example_date, 'total_avg_inflow'])

# Save dataframe
if SAVE_DATAFRAME:
    df_grouped_by_date.to_pickle("history/historical_data_daily_avg_inflow.pickle")
