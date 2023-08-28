# Data: history

This folder should contain the following files:
- ``historical_data.pickle``: Historical data, a pickled Pandas data frame
with the time-varying data (e.g. price of energy, observed reservoir volumes).
- ``historical_data_reliable_only.pickle``: Alternative version of the previous file,
in which unregulated flows NANs where removed (instead of set to 0).
This file is rarely used in this project.

After executing ``../create_historical_data_clean.py``,
these additional files will be created in this folder:
- ``historical_data_clean.pickle``: Historical data with no duplicates and
no missing datetimes.
- ``historical_data_clean_train.pickle``: The first 80% of rows of the previous file.
- ``historical_data_clean_test.pickle``: The remaining 20% of rows.

And after executing ``../create_historical_data_daily_avg_inflow.py``,
the file ``historical_data_daily_avg_inflow.pickle`` will be created,
with contains the total average inflow of each day.

All of these files are git-ignored to avoid occupying too much space in the repository.
