import pandas as pd

# Read training data
path_training_data = "../data/rl_training_data/training_data.pickle"
training_data = pd.read_pickle(path_training_data)
print(training_data)

# Create training data with no duplicates
training_data_no_duplicates = training_data.drop_duplicates(subset="datetime")
training_data_no_duplicates.reset_index(inplace=True, drop=True)
training_data_no_duplicates.to_pickle("../data/rl_training_data/training_data_no_duplicates.pickle")
print(training_data_no_duplicates)
