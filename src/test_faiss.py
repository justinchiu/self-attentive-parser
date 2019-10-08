import torch
import numpy as np
import faiss

print(faiss.__version__)

np.random.seed(1234)             # make reproducible
d = 256
nb = int(1e4)
nq = 50
nlist = 100 
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained

xb = np.random.random((nb // 2, d)).astype('float32')
xb[:, 0] += np.arange(xb.shape[0]) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index.train(xb)
index.add(xb)

res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
