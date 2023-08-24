# Training data

This folder should contain the data with which the
reinforcement learning (RL) agent will be trained:
* `constants_Ndams.json`, with the data that remains constant throughout different episodes
  (e.g. minimum volume of each dam).
* `historical_data.pickle`, a pickled Pandas data frame with the time-varying data
  (e.g. price of energy).

The second file is git-ignored to avoid occupying too much space in the repository.
