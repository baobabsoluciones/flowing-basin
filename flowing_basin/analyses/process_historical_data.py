import pandas as pd
from datetime import datetime, timedelta

# Read historical data ---- #

path_data = "../data/rl_training_data/historical_data.pickle"
data = pd.read_pickle(path_data)
print(data)

# Create historical data with no duplicates ---- #

data_no_duplicates = data.drop_duplicates(subset="datetime")
data_no_duplicates.reset_index(inplace=True, drop=True)
print(data_no_duplicates)

# Create historical data with no duplicates AND all datetimes (CLEAN data) ---- #

# List of all datetimes
start_date = data.loc[0, "datetime"]
end_date = data.loc[data.shape[0] - 1, "datetime"]
print(start_date, end_date)
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += timedelta(minutes=15)
print(dates[0:15])
print(data_no_duplicates.loc[:, "datetime"].tolist()[0:15])

# Add the missing datetimes to a new dataframe by copying adjacent rows
data_all_dates = data_no_duplicates.copy()
date_index = 0
for index, row in data_no_duplicates.iterrows():
    current_date = dates[date_index]
    actual_date = row["datetime"]
    if current_date != actual_date:
        # There are missing datetimes
        # Let's add them by copying the row:
        while current_date < actual_date:

            # Row with the missing datetime
            new_row = row.copy()
            new_row["datetime"] = current_date
            print(date_index, current_date, actual_date, index)

            # Add row to the new dataframe
            data_all_dates.loc[date_index - 0.5] = new_row  # New row between the current row and the next one
            data_all_dates = data_all_dates.sort_index().reset_index(drop=True)
            date_index += 1
            current_date += timedelta(minutes=15)
    date_index += 1
data_all_dates.reset_index(inplace=True, drop=True)
assert dates == data_all_dates.loc[:, "datetime"].tolist()
print(data_all_dates)
data_all_dates.to_pickle("../data/rl_training_data/historical_data_clean.pickle")

# Split clean historical data into train and test data ---- #

total_rows = data_all_dates.shape[0]
train_size = int(0.8 * total_rows)
train_data = data_all_dates[:train_size]
train_data.to_pickle("../data/rl_training_data/historical_data_clean_train.pickle")
print(train_data)

test_data = data_all_dates[train_size:]
test_data.reset_index(inplace=True, drop=True)
test_data.to_pickle("../data/rl_training_data/historical_data_clean_test.pickle")
print(test_data)
