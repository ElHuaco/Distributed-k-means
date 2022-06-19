import numpy as np
import dask
import dask.array as da
import dask_ml

def lloyd_scalable (X, k, centroids = None, maxIter = 1000, patience = 1e-6):
    X = make_da(X)
    n_points, n_features = X.shape
    if centroids is None:
        random_index = np.random.choice(n_points, size=(1, k), replace=False)
        centroids = X[random_index[0]]
    epoch = 1
    loss_diff = patience + 1.
    loss = 0
    while (epoch < maxIter and loss_diff > patience):
        distances_matrix = dask_ml.metrics.pairwise_distances(X, centroids)
        indeces = da.argmin(distances_matrix, axis=1)
        new_loss = distances_matrix[indeces].sum()
        # Possible bug if new_centroids = zeros and we don't enter the for loop
        new_centroids = da.zeros((k, n_features))
        print(indeces.max())
        for i in range(indeces.max()):
            new_centroids[i] = X[indeces == i].mean(axis=0)
        epoch = epoch + 1
        loss_diff = da.absolute(new_loss - loss)
        centroids = new_centroids
        loss = new_loss
    return (centroids, indeces)
