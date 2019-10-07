
import index
import torch

import numpy as np

index_path = "index"
model_base_path = "models/en_bert_empty_dev=95.57.pt"
prefix = index.get_index_prefix(
    index_base_path = index_path,
    full_model_path = model_base_path,
    nn_prefix = "all_spans_empty",
)

annoy_index = index.AnnoyIndex()
faiss_index = index.FaissIndex()

annoy_index.load(prefix)
print("loaded annoy")
prefix = index.get_index_prefix(
    index_base_path = index_path,
    full_model_path = model_base_path,
    nn_prefix = "all_spans_empty_test",
)
faiss_index.load(prefix)
print("loaded faiss")

T = 40
H = 250
L = 113

# torch stuff
chart = torch.randn(T+1, T+1, L)
reps = torch.randn(T+1, T+1, H)
indices = torch.LongTensor([
    (left, left+length)
    for length in range(1, T+1)
    for left in range(0, T+1-length)
])
left = indices[:,0]
right = indices[:,1]

# numpy stuff
chart_np = chart.numpy()
reps_np = reps.numpy()
indices_np = indices.numpy()
left_np = indices_np[:,0]
right_np = indices_np[:,1]

### TOPK TESTS

# cpu test
## numpy
def annoy_np():
    c = chart_np.copy()
    r = reps_np[left_np, right_np]
    labels, distances = annoy_index.topk(r, 8)
    for le, ri, l, d in zip(
        left_np, right_np,
        labels[0], distances[0],
    ):
        np.logaddexp.at(c[le, ri], l, d)
    return c, labels, distances

def faiss_np():
    c = chart_np.copy()
    r = reps_np[left_np, right_np]
    labels, distances = faiss_index.topk(r, 8)
    for le, ri, l, d in zip(
        left_np, right_np,
        labels[0], distances[0],
    ):
        np.logaddexp.at(c[le, ri], l, d)
    return c, labels, distances

## torch
def faiss():
    c = chart.clone()
    r = reps[left, right]

# gpu test


# correctness
ca, la, da = annoy_np()
cf, lf, df = faiss_np()
#import pdb; pdb.set_trace()
# They're different!

import time
def get_time(f, K=5):
    t = time.time()
    _ = [f() for _ in range(K)]
    return (time.time() - t)

print(get_time(annoy_np))
print(get_time(faiss_np))

import pdb; pdb.set_trace()
