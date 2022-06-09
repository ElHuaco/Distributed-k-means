# Distributed-k-means

- ## Why did we choose Dask VS Spark?
  - https://docs.dask.org/en/stable/spark.html#summary
- ## How can we download in a distributed manner the dataset?
  - ``dask_ml.datasets`` package.
  - ``dask.bags`` package: https://docs.dask.org/en/latest/bag-creation.html
- ## What is the Dask k-means|| API?
Below is a Dask _k-means||_ API snippet, extracted from the [dask examples webpage](https://examples.dask.org/machine-learning/training-on-large-datasets.html?highlight=k%20means). It uses the ``dask_ml.cluster`` package.
``` python
X, y = dask_ml.datasets.make_blobs(n_samples=1000000,
                                   chunks=100000,
                                   random_state=0,
                                   centers=3)
X = X.persist()
km = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=2, oversampling_factor=10)
km.fit(X)
```
