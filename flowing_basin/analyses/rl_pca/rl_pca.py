import numpy as np
from sklearn.decomposition import PCA

# array of shape (n_samples, n_features)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = np.load("observations_data/observationsO2.npy")
print("Observation record shape:", X.shape)
print("Observation record:", X)

pca = PCA(n_components=1)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
