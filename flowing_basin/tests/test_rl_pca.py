import numpy as np
from sklearn.decomposition import PCA
import os

OBSERVATION_TYPE = "O2"
OBSERVATIONS_FOLDER = f"reports/observations_data/observations{OBSERVATION_TYPE}"
MODEL_FILE = f"reports/observations_data/observations{OBSERVATION_TYPE}_model.sav"
EXPLAINED_VARIANCE = 0.99

# Input data ---- #
# array of shape (n_samples, n_features)
X = np.load(os.path.join(OBSERVATIONS_FOLDER, 'observations.npy'))
print("Observation record shape:", X.shape)
print("Observation record:", X)

# Train model ---- #
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
