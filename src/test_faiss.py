import torch
import numpy as np
import faiss

np.random.seed(1234)             # make reproducible

d = 256
nb = int(1e4)
nq = 50
nlist = 128 # slightly more than the number of labels
#quantizer = faiss.IndexFlatIP(d)
#index = faiss.IndexIVFFlat(quantizer, d, nlist)
index = faiss.index_factory(d, "PCAR256,IVF128,Flat", faiss.METRIC_INNER_PRODUCT)
assert not index.is_trained
#print(index.is_trained)

# need to pack queries

xb = np.random.random((nb // 2, d)).astype('float32')
xb[:, 0] += np.arange(xb.shape[0]) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

#print(index.is_trained)
# why is this true lol
index.train(xb)
assert index.is_trained

index.add(xb)
print(index.ntotal)

"""
xb2 = np.random.random((nb // 2, d)).astype('float32')
xb2[:, 0] += np.arange(xb.shape[0]) / 1000.
index.add(xb2)
print(index.ntotal)
"""

# search
k = 8
D, I = index.search(xb[:5], k)
print(I)
print(D)

D, I = index.search(xb[-5:], k)
print(I)
print(D)

index.nprobe = 4


faiss.write_index(index, "faisstest.ann")
index0 = faiss.read_index("faisstest.ann")
D0, I0 = index.search(xb[-5:], k)

assert np.allclose(D, D0)
assert np.allclose(I, I0)
