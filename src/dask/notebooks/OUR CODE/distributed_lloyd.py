import numpy as np
import dask
import dask.array as da
import dask_ml

def lloyd_scalable (X, k, centroids = None, maxIter = 1000, patience = 1e-6):
    if centroids is None:
        random_index = np.random.choice(len(X), size=(1, k), replace=False)
        centroids = X[random_index[0]]
    epoch = 1
    len_X = X.shape[1]
    while (epoch < maxIter):
        indeces = da.argmin(dask_ml.metrics.pairwise_distances(X, np.array(centroids)), axis=1)
        new_centroids = da.zeros((k, len_X))
        for i in range(indeces.max()):
            new_centroids[i] = X[indeces == i].mean(axis=0)
        epoch = epoch + 1
        centroids = new_centroids
    return new_centroids.compute()
