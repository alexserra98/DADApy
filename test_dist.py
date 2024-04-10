import numpy as np
from dadapy.data import Data
import time

rand_mat = np.random.rand(1400, 4000)
data = Data(rand_mat)
#track elapsed time
start = time.time()
data.compute_distances( metric='cosine')
end = time.time()
print('Elapsed time with cosine: ', end - start)
start = time.time()
data.compute_distances()
end = time.time()
print('Elapsed time standard: ', end - start)
