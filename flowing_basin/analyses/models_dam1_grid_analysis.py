from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import numpy as np
from mayavi import mlab

# Load model of first dam
paths_power_model = "../ml_models/model_E1.sav"
power_model = PowerGroup.get_nonlinear_power_model(paths_power_model)

# Get necessary data of first dam
instance = Instance.from_json("../data/rl_training_data/constants.json")
turbined_flow_points = instance.get_turbined_flow_obs_for_power_group("dam1")
max_flow = instance.get_max_flow_of_channel("dam1")

# Axes of surface plot
lags1 = np.linspace(0, max_flow, num=100)
lags2 = np.linspace(0, max_flow, num=100)
lags1_mesh, lags2_mesh = np.meshgrid(lags1, lags2)
lags1_mesh_t = np.transpose(lags1_mesh)
lags2_mesh_t = np.transpose(lags2_mesh)
print("Lags 1:", lags1_mesh_t.shape, lags1_mesh_t)
print("Lags 2:", lags2_mesh_t.shape, lags2_mesh_t)

# Turbined flows ---- #

# Turbined flows given by ML model
lags1_mesh_t_f = lags1_mesh_t.flatten()
lags2_mesh_t_f = lags2_mesh_t.flatten()
model_input = np.transpose(np.array([lags1_mesh_t_f, lags2_mesh_t_f]))
ml_powers = power_model.predict(model_input)
ml_turbined_flows = np.interp(
    ml_powers,
    turbined_flow_points["observed_powers"],
    turbined_flow_points["observed_flows"],
)
ml_turbined_flows_mesh = ml_turbined_flows.reshape(lags1_mesh_t.shape)
print("ML turbined, max:", ml_turbined_flows.max())
print("ML turbined:", ml_turbined_flows_mesh.shape, ml_turbined_flows_mesh)

# Linear turbined flows
linear_turbined_flows_mesh = lags1_mesh_t
print("Linear turbined:", linear_turbined_flows_mesh.shape, linear_turbined_flows_mesh)

# Compare
ml_turbined_surface = mlab.surf(lags1_mesh_t, lags2_mesh_t, ml_turbined_flows_mesh)
linear_turbined_surface = mlab.surf(lags1_mesh_t, lags2_mesh_t, linear_turbined_flows_mesh, color=(0.6, 0.6, 0.6), opacity=0.75)
mlab.axes(xlabel="lag1", ylabel="lag2", zlabel="Turbined flow")
# mlab.savefig(filename='ml_models_analyses_files/model_E1_graph.png', size=(2000, 2000))  # <- Doesn't work
mlab.show()
mlab.clf()

# Powers ---- #

# Powers given by ML model
ml_powers_mesh = ml_powers.reshape(lags1_mesh_t.shape)
print("ML powers:", ml_turbined_flows_mesh.shape, ml_turbined_flows_mesh)

# Powers given by linear model
linear_turbined_flows = lags1_mesh_t_f
linear_powers = np.interp(
    linear_turbined_flows,
    turbined_flow_points["observed_flows"],
    turbined_flow_points["observed_powers"],
)
linear_powers_mesh = linear_powers.reshape(lags1_mesh_t.shape)
print("Linear powers:", linear_powers_mesh.shape, linear_powers_mesh)

# Compare
ml_power_surface = mlab.surf(lags1_mesh_t, lags2_mesh_t, ml_powers_mesh, warp_scale=2.)
linear_power_surface = mlab.surf(lags1_mesh_t, lags2_mesh_t, linear_powers_mesh, warp_scale=2., color=(0.6, 0.6, 0.6), opacity=0.75)
mlab.axes(xlabel="lag1", ylabel="lag2", zlabel="Power", ranges=[0, max_flow, 0, max_flow, 0, ml_powers.max()])
mlab.show()

