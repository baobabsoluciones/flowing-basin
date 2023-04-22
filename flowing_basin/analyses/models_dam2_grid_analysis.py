from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np
import pandas as pd

# Load model of second dam
paths_power_model = "../ml_models/model_E2.sav"
power_model = PowerGroup.get_power_model(paths_power_model)

# Get necessary data of second dam
instance = Instance.from_json("../data/rl_training_data/constants.json")
turbined_flow_points = instance.get_turbined_flow_obs_for_power_group("dam2")
max_flow = instance.get_max_flow_of_channel("dam2")

# Input lags
num_lags = 10
lags3 = np.linspace(0, max_flow, num=num_lags)
lags4 = np.linspace(0, max_flow, num=num_lags)
lags5 = np.linspace(0, max_flow, num=num_lags)
lags6 = np.linspace(0, max_flow, num=num_lags)
lags3_mesh, lags4_mesh, lags5_mesh, lags6_mesh = np.meshgrid(lags3, lags4, lags5, lags6)
print("Lags 3:", lags3_mesh.shape, lags3_mesh)

# Powers given by ML model
lags3_mesh_f = lags3_mesh.flatten()
lags4_mesh_f = lags4_mesh.flatten()
lags5_mesh_f = lags5_mesh.flatten()
lags6_mesh_f = lags6_mesh.flatten()
model_input = np.transpose(np.array([lags3_mesh_f, lags4_mesh_f, lags5_mesh_f, lags6_mesh_f]))
print("ML model input:", model_input)
ml_powers = power_model.predict(model_input)
ml_powers_mesh = ml_powers.reshape(lags3_mesh.shape)
print("ML model powers:", ml_powers_mesh.shape, ml_powers_mesh)

# Very different values for similar lags within the ML power values ---- #

# model_input_mesh1, model_input_mesh2 = np.meshgrid(model_input, model_input)
# lag_differences = model_input_mesh1 - model_input_mesh2
# print("Lag differences:", lag_differences.shape, lag_differences)
# small_differences = np.abs(lag_differences) < 0.75
# This doesn't work; it needs more than 11.9 GiB of memory
# TODO: do this analysis

# Differences between ML and linear power values ---- #

# Powers given by linear model
linear_turbined_flows = (lags3_mesh_f + lags4_mesh_f + lags5_mesh_f) / 3
linear_powers = np.interp(
    linear_turbined_flows,
    turbined_flow_points["observed_flows"],
    turbined_flow_points["observed_powers"],
)
linear_powers_mesh = linear_powers.reshape(lags3_mesh.shape)
print("Linear powers:", linear_powers_mesh.shape, linear_powers_mesh)

# Absolute differences
differences = ml_powers_mesh - linear_powers_mesh
print("Differences:", differences)

# Filter
# 1. Consider only dramatic differences
threshold = 1.5
mask1 = np.abs(differences) > threshold
# 2. Consider only smooth combinations of lags (k=1)
var_lags56 = lags5_mesh - lags6_mesh
var_lags45 = lags4_mesh - lags5_mesh
var_lags34 = lags3_mesh - lags4_mesh
mask2 = np.logical_not(
    np.logical_or(
        var_lags56 * var_lags45 < 0,
        var_lags45 * var_lags34 < 0,
    )
)
mask = mask1 & mask2
df = pd.DataFrame()
df["power_ml"] = ml_powers_mesh[mask]
df["power_linear"] = linear_powers_mesh[mask]
df["lag3"] = lags3_mesh[mask]
df["lag4"] = lags4_mesh[mask]
df["lag5"] = lags5_mesh[mask]
df["lag6"] = lags6_mesh[mask]
print(df)
