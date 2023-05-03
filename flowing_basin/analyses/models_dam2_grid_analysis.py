from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np
import pandas as pd

# Load model of second dam
paths_power_model = "../ml_models/model_E2.sav"
power_model = PowerGroup.get_nonlinear_power_model(paths_power_model)

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
print("ML model input:", model_input.shape, model_input)
ml_powers = power_model.predict(model_input)
ml_powers_mesh = ml_powers.reshape(lags3_mesh.shape)
print("ML model powers:", ml_powers_mesh.shape, ml_powers_mesh)

# ML model discontinuities: very different values for similar lags within the ML power values ---- #

lags_threshold = 2
power_threshold = 1

# 1. Get the combinations of lags that are very similar (difference below threshold)

lag_diffs = model_input[:, np.newaxis] - model_input
# Here, we add a singleton dimension to model_input, so when we subtract the original model_input from it, both arrays
# are copied along the first and second axes, respectively, providing all posible differences between lag combinations:
# model_input[:, np.newaxis] (3d array): N x 1 x 4
# model_input                (2d array):     N x 4
# Broadcasting result        (3d array): N x N x 4
# Where N is the number of lag combinations.
# Source: https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcastable-arrays

lags_mask = np.sum(np.abs(lag_diffs), axis=2) < lags_threshold
# Result (2d array): N x N

# 2. Get the combinations of lags that are smooth (k=1)

var_lags56 = model_input[:, 2] - model_input[:, 3]
var_lags45 = model_input[:, 1] - model_input[:, 2]
var_lags34 = model_input[:, 0] - model_input[:, 1]
smooth_lags = np.logical_not(
    np.logical_or(
        var_lags56 * var_lags45 < 0,
        var_lags45 * var_lags34 < 0,
    )
)

smooth_mask = smooth_lags[:, np.newaxis] * smooth_lags
# smooth_lags[:, np.newaxis] (2d array): N x 1
# smooth_lags                (1d array):     N
# Broadcasting result        (2d array): N x N

# 3. Get the power values that are very different (difference above threshold)

power_diffs = ml_powers[:, np.newaxis] - ml_powers
# ml_powers[:, np.newaxis] (2d array): N x 1
# ml_powers                (1d array):     N
# Broadcasting result      (2d array): N x N

power_mask = np.abs(power_diffs) > power_threshold
# Result (2d array): N x N

# 4. Take the lags and power values where lags are similar and smooth, but power values are different

full_mask = lags_mask & smooth_mask & power_mask
filtered_rows, filtered_cols = full_mask.nonzero()

df_discont = pd.DataFrame({
    "case1_power_ml": ml_powers[filtered_rows],
    "case1_lag3": model_input[filtered_rows, 0],
    "case1_lag4": model_input[filtered_rows, 1],
    "case1_lag5": model_input[filtered_rows, 2],
    "case1_lag6": model_input[filtered_rows, 3],
    "case2_power_ml": ml_powers[filtered_cols],
    "case2_lag3": model_input[filtered_cols, 0],
    "case2_lag4": model_input[filtered_cols, 1],
    "case2_lag5": model_input[filtered_cols, 2],
    "case2_lag6": model_input[filtered_cols, 3]
})
print("Discontinuities:", df_discont.to_string())
df_discont.to_excel("ml_models_analyses_files/dam2_MLModelDiscontinuities.xlsx")

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
mask1_mesh = np.abs(differences) > threshold
# 2. Consider only smooth combinations of lags (k=1)
var_lags56_mesh = lags5_mesh - lags6_mesh
var_lags45_mesh = lags4_mesh - lags5_mesh
var_lags34_mesh = lags3_mesh - lags4_mesh
mask2_mesh = np.logical_not(
    np.logical_or(
        var_lags56_mesh * var_lags45_mesh < 0,
        var_lags45_mesh * var_lags34_mesh < 0,
    )
)
mask_mesh = mask1_mesh & mask2_mesh
df_diffs = pd.DataFrame()
df_diffs["power_ml"] = ml_powers_mesh[mask_mesh]
df_diffs["power_linear"] = linear_powers_mesh[mask_mesh]
df_diffs["lag3"] = lags3_mesh[mask_mesh]
df_diffs["lag4"] = lags4_mesh[mask_mesh]
df_diffs["lag5"] = lags5_mesh[mask_mesh]
df_diffs["lag6"] = lags6_mesh[mask_mesh]
# print("Differences:", df_diffs.to_string())
df_diffs.to_excel("ml_models_analyses_files/dam2_MLvsLinearModel.xlsx")
