# Distributed-k-means

- ## Why did we choose Dask VS Spark?
- ## How can we download in batches the dataset to have a sensical distributed operation?
- ## What is the Dask k-means|| API?
Below is a Dask _k-means||_ API snippet, extracted from the [dask examples webpage](https://examples.dask.org/machine-learning/training-on-large-datasets.html?highlight=k%20means).
``` python
import dask_ml.cluster
import dask_ml.datasets
# Scale up: increase n_samples or n_features
X, y = dask_ml.datasets.make_blobs(n_samples=1000000,
                                   chunks=100000,
                                   random_state=0,
                                   centers=3)
X = X.persist()
km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)
```
