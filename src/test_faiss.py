import torch
import numpy as np
import faiss

import time
def get_time(t):
    return time.time() - t

print(faiss.__version__)

np.random.seed(1234)             # make reproducible
K = 8
draw = 1024
d = 256
nb = int(1e7)
nq = 64

# setup
xb = np.random.random((nb // 2, d)).astype('float32')
xb[:, 0] += np.arange(xb.shape[0]) / 1000.

xq = xb[:32]
#xq = np.random.random((nq, d)).astype('float32')
#xq[:, 0] += np.arange(nq) / 1000.

def test_time(index, xb, xq, K, eD=None, eI=None):
    t = time.time()
    index.train(xb)
    index.add(xb)
    print(f"Training and adding: {get_time(t)}s")
    t = time.time()
    D, I = index.search(xq, K)
    print(f"Query: {get_time(t)}s")
    if eI is not None:
        # calculate PR later
        overlap = (eI == I).sum()
        print(f"Overlapping idxs: {overlap}")
    return D, I

# indices

bigass_index = faiss.IndexFlatIP(d)
eD, eI = test_time(bigass_index, xb, xq, K)

bigass_index_l2 = faiss.IndexFlatL2(d)
eD2, eI2 = test_time(bigass_index_l2, xb, xq, K)

nlist = 100 
quantizer = faiss.IndexFlatIP(d)
cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

quantizer = faiss.IndexFlatL2(d)
cpu_index_l2 = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

#index = faiss.index_factory(d, "IVF4096,PQ64", faiss.METRIC_INNER_PRODUCT)
index = faiss.index_factory(d, "IVF4096,PQ32", faiss.METRIC_L2)
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 3, index)

print(cpu_index)
#test_time(cpu_index, xb, xq, K, eD, eI)
#cpu_index.nprobe = 10
#test_time(cpu_index, xb, xq, K, eD, eI)
cpu_index.nprobe = 100
test_time(cpu_index, xb, xq, K, eD, eI)
print(cpu_index_l2)
test_time(cpu_index_l2, xb, xq, K, eD, eI)
#print(index)
#test_time(index, xb, xq, K, eD, eI)
print(index_gpu)
test_time(index_gpu, xb, xq, K, eD, eI)
