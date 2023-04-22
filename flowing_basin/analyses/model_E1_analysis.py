from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np

# Load model of first dam
paths_power_model = "../ml_models/model_E1.sav"
power_model = PowerGroup.get_power_model(paths_power_model)

# Get necessary data of first dam
instance = Instance.from_json("../data/rl_training_data/constants.json")
turbined_flow_points = instance.get_turbined_flow_obs_for_power_group("dam1")

# Get turbined flow for one particular set of lags
lag1 = 5.4
lag2 = 3.4
power = power_model.predict(np.array([lag1, lag2]).reshape(1, -1))
turbined_flow = np.interp(
    power,
    turbined_flow_points["observed_powers"],
    turbined_flow_points["observed_flows"],
)
print(turbined_flow)
