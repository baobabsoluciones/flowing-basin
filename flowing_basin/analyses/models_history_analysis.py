from flowing_basin.core import Instance
from flowing_basin.tools import PowerGroup
import pandas as pd
import numpy as np

# Instance
instance = Instance.from_json("../data/rl_training_data/constants.json")

# History
path_historical_data = "../data/rl_training_data/historical_data_reliable_only.pickle"
df_history = pd.read_pickle(path_historical_data)
print(df_history)

for dam in [1, 2]:

    print(f"---- dam{dam} ----")

    # Load model of dam
    paths_power_model = f"../ml_models/model_E{dam}.sav"
    power_model = PowerGroup.get_nonlinear_power_model(paths_power_model)

    # Get necessary data of dam
    turbined_flow_points = instance.get_turbined_flow_obs_for_power_group(f"dam{dam}")

    # Relevant columns from history
    df_comparison = pd.DataFrame()
    df_comparison["flow"] = df_history[f"dam{dam}_flow"]
    relevant_lags = instance.get_relevant_lags_of_dam(f"dam{dam}")
    for lag in relevant_lags:
        df_comparison[f"lag{lag}"] = np.roll(df_history[f"dam{dam}_flow"], lag)
    df_comparison["power_real"] = df_history[f"dam{dam}_power"]
    df_comparison.drop(
        index=df_comparison.index[[i for i in range(relevant_lags[-1])]], axis=0, inplace=True
    )  # Since the first N rows of the data frame do not make sense

    # Powers predicted by ML model
    df_comparison["power_ML"] = power_model.predict(df_comparison[[f"lag{lag}" for lag in relevant_lags]])

    # Powers predicted by linear model
    verification_lags = instance.get_verification_lags_of_dam(f"dam{dam}")
    linear_turbined_flows = df_comparison[[f"lag{lag}" for lag in verification_lags]].mean(axis=1)
    linear_powers = np.interp(
        linear_turbined_flows,
        turbined_flow_points["observed_flows"],
        turbined_flow_points["observed_powers"],
    )
    df_comparison["power_linear"] = linear_powers
    print(df_comparison)

    # Get Mean Square Error of linear and ML model predictions
    mse_linear = ((df_comparison["power_real"] - df_comparison["power_linear"])**2).mean()
    mse_ml = ((df_comparison["power_real"] - df_comparison["power_ML"])**2).mean()
    print(f"Mean square errors: {mse_linear=}, {mse_ml=}")
