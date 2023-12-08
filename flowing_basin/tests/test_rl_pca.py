import numpy as np
from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os

OBSERVATION_TYPE = "O1"
OBSERVATIONS_FOLDER = f"reports/observations_data/observations{OBSERVATION_TYPE}"
CONSTANTS = "../data/constants/constants_2dams.json"
HISTORICAL_DATA = "../data/history/historical_data_clean.pickle"
OBSERVATIONS_JSON = f"reports/observations_data/observations{OBSERVATION_TYPE}/config.json"
# MODEL_FILE = f"reports/observations_data/observations{OBSERVATION_TYPE}_model.sav"
EXPLAINED_VARIANCE = 0.99

# Input data ---- #
# array of shape (n_samples, n_features)
X = np.load(os.path.join(OBSERVATIONS_FOLDER, 'observations.npy'))
print("Observation record shape:", X.shape)
print("Observation record:", X)

# Simple data analysis
constants = Instance.from_json(CONSTANTS)
config = RLConfiguration.from_json(OBSERVATIONS_JSON)
projector = Projector.from_config(config, OBSERVATIONS_FOLDER)
env = RLEnvironment(
    config=config,
    projector=projector,
    path_constants=CONSTANTS,
    path_historical_data=HISTORICAL_DATA,
)
averages = np.mean(X, axis=0)
indices = env.get_observation_indices(flattened=True)
print(indices)
for dam_id in constants.get_ids_of_dams():
    max_sight = max(config.num_steps_sight[feature, dam_id] for feature in config.features)
    num_features = len(config.features)
    fig, axs = plt.subplots(max_sight, num_features)
    fig.suptitle(f"Histograms of features for {dam_id}")
    for feature_index, feature in enumerate(config.features):
        for lookback in range(config.num_steps_sight[feature, dam_id]):
            ax = axs[lookback, feature_index]
            if constants.get_order_of_dam(dam_id) == 1 or feature not in config.unique_features:
                index = indices[dam_id, feature, lookback]
                print(f"{dam_id=} {feature=} {lookback=} | {averages[index]}")
                ax.hist(X[:, index], bins='auto')
            ax.set_yticklabels([])  # Hide y-axis tick labels
            ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
            if lookback == 0:
                ax.set_title(feature)
    plt.tight_layout()
    plt.show()


# Train PCA model ---- #
# n_components can also be a fraction (e.g., 0.85) to indicate the desired explained variance
pca = PCA(n_components=EXPLAINED_VARIANCE)
pca.fit(X)
print("Explained variance of each component:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print("Number of components:", pca.n_components_)

# # Save the model to disk
# pickle.dump(pca, open(MODEL_FILE, 'wb'))
# print(f"Saved model to {MODEL_FILE}")

# Transformed input data ---- #
print("Original data:", X[0, :])
print("Original data minimum values:", list(np.min(X, axis=0)))
print("Original data maximum values:", list(np.max(X, axis=0)))
X_transformed = pca.transform(X)
print("Transformed data:", X_transformed[0, :])
print("Transformed data minimum values:", list(np.min(X_transformed, axis=0)))
print("Transformed data maximum values:", list(np.max(X_transformed, axis=0)))
print("Transformed data minimum value:", np.min(X_transformed))
print("Transformed data maximum value:", np.max(X_transformed))
print("Inverse transform:", pca.inverse_transform(X_transformed)[0, :])