# Distributed-k-means

- ## Why did we choose Dask VS Spark?
  - https://docs.dask.org/en/stable/spark.html#summary
- ## How can we download in a distributed manner the dataset?
  - ``dask_ml.datasets`` package if it's defined
  - ``dask.bags`` package: https://docs.dask.org/en/latest/bag-creation.html
  - ``skelearn.datasets`` to a dask dataframe with ``dask.dataframes``: https://docs.dask.org/en/stable/dataframe-create.html
- ## What optimizations have we considered in the implementation?
  - Once $|x_i-c_j|^2$ is computed for a $(x_i \in X,\text{ } c_j \in C)$ pair, and taken into $\phi_X(C)$, we don't need to compute it again. Moreover, we only need to store it if it's the minimun distance so far.
  - TODO: Check if we needed O(1) instead of O($\log\phi$) in the end
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
