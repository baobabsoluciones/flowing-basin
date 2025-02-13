from flowing_basin.core import Instance
from cornflow_client.core.tools import load_json
import pandas as pd

path_constants = "../../data/constants/constants_2dams.json"
constants = load_json(path_constants)
inst_const = Instance.from_dict(constants)

path_historical_data = "../../data/history/historical_data.pickle"
historical_data = pd.read_pickle(path_historical_data)
print(historical_data.head().to_string())

# Prices
print(
    "prices (mean, min, max): ",
    historical_data["price"].mean(),
    historical_data["price"].min(),
    historical_data["price"].max(),
)

# Power installed
print(
    "power installed (dam1, dam2): ",
    inst_const.get_max_power_of_power_group("dam1"),
    inst_const.get_max_power_of_power_group("dam2")
)
