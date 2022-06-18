import time
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

our = np.array([])
for i in range(2, 200):    
    x = np.random.uniform(0, 10, (i, 5))
    y = np.random.uniform(0, 10, (i, 5))
    start = time.time()
    OUR_pairwise_distance(x, y)
    our = np.append(our, time.time()-start)
    
    
sklearn = np.array([])
for i in range(2, 200):    
    x = np.random.uniform(0, 10, (i, 5))
    y = np.random.uniform(0, 10, (i, 5))
    start = time.time()
    metrics.pairwise_distances(x, y, n_jobs=-1)
    sklearn = np.append(sklearn, time.time()-start)
    
    
x = np.arange(2, 200)*5
plt.figure(figsize=(5, 3), dpi=200)
plt.title('Pairwise distance performance')
plt.plot(x, our, 'red', label='Our')
plt.plot(x, sklearn, 'blue', label='Sklearn')
plt.xlabel('N. of values per Array')
plt.ylabel('Time to calculate')
plt.legend()
plt.show()