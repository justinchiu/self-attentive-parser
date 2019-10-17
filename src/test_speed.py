import time

import index
import torch
import scatter

import numpy as np

index_path = "index"
model_base_path = "models/en_bert_empty_nl2_dev=95.36.pt"
prefix = index.get_index_prefix(
    index_base_path = index_path,
    full_model_path = model_base_path,
    nn_prefix = "all_spans_empty_nl2",
)

annoy_index = index.AnnoyIndex(metric="l2")
faiss_index = index.FaissIndex(metric="l2")
faiss_index_gpu = index.FaissIndex(metric="l2")

t = time.time()
annoy_index.load(prefix)
print(f"loaded annoy {time.time() - t}")
t = time.time()
prefix = index.get_index_prefix(
    index_base_path = index_path,
    full_model_path = model_base_path,
    nn_prefix = "all_spans_empty_nl2",
)
faiss_index.load(prefix)
print(f"loaded faiss {time.time() - t}")
t = time.time()
faiss_index_gpu.load(prefix)
faiss_index_gpu.to(0)
print(f"loaded faiss gpu {time.time() - t}")

T = 40
H = 256
L = 113

# torch stuff
chart = torch.zeros(T+1, T+1, L)
reps = torch.randn(T+1, T+1, H)
indices = torch.LongTensor([
    (left, left+length)
    for length in range(1, T+1)
    for left in range(0, T+1-length)
])
left = indices[:,0]
right = indices[:,1]
flat_indices = left * (T+1) + right

# numpy stuff
chart_np = chart.numpy()
reps_np = reps.numpy()
indices_np = indices.numpy()
left_np = indices_np[:,0]
right_np = indices_np[:,1]

# gpu
chart_gpu = chart.cuda()
reps_gpu = reps.cuda()
flat_indices_gpu = flat_indices.cuda()

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
        c[le, ri, l] = float("-inf")
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
        c[le, ri, l] = float("-inf")
        np.logaddexp.at(c[le, ri], l, d)
    return c, labels, distances

## torch then np aggregation
def faiss_t_np():
    c = chart_np.copy()
    r = reps[left, right]
    labels, distances = faiss_index.topk_torch_np(r, 8)
    for le, ri, l, d in zip(
        left_np, right_np,
        labels[0], distances[0],
    ):
        c[le, ri, l] = float("-inf")
        np.logaddexp.at(c[le, ri], l, d)
    return c, labels, distances

def faiss_t_t():
    c = chart.clone()
    flat_chart = c.view(-1, c.shape[-1])
    r = reps[left, right]
    labels, distances = faiss_index.topk_torch(r, 8)
    cells = scatter.scatter_lse(
        distances[0],
        labels[0],
        dim = -1,
        dim_size = L,
        fill_value = 0,
    )
    flat_chart = torch.zeros((T+1) * (T+1), L)
    flat_chart = flat_chart.scatter(
        0,
        flat_indices.unsqueeze(-1).expand_as(cells),
        cells,
    )
    c = (flat_chart
        .view(T+1, T+1, flat_chart.shape[-1])
    )
    return (
        c.cpu().numpy(),
        labels.cpu().numpy(),
        distances.cpu().numpy(),
    )

# gpu test
def faiss_gpu():
    c = chart_gpu.clone()
    flat_chart = c.view(-1, c.shape[-1])
    r = reps_gpu[left, right]
    labels, distances = faiss_index_gpu.topk_torch(r, 8)
    cells = scatter.scatter_lse(
        distances[0],
        labels[0],
        dim = -1,
        dim_size = L,
        fill_value = 0,
    )
    flat_chart = torch.zeros((T+1) * (T+1), L, device=c.device)
    flat_chart = flat_chart.scatter(
        0,
        flat_indices_gpu.unsqueeze(-1).expand_as(cells),
        cells,
    )
    c = (flat_chart
        .view(T+1, T+1, flat_chart.shape[-1])
    )
    return (
        c.cpu().numpy(),
        labels.cpu().numpy(),
        distances.cpu().numpy(),
    )

"""
# correctness
ca, la, da = annoy_np()
cf, lf, df = faiss_np()
#import pdb; pdb.set_trace()
# They're different!
cf0, lf0, df0 = faiss_t_np()
assert np.allclose(cf, cf0)
assert np.allclose(lf, lf0)
assert np.allclose(df, df0)
c0, l0, d0 = faiss_t_t()
assert np.allclose(lf, l0)
assert np.allclose(df, d0)
#import pdb; pdb.set_trace()
assert np.allclose(cf, c0)
"""

def get_time(f, K=5):
    t = time.time()
    _ = [f() for _ in range(K)]
    return (time.time() - t)

print("annoy_np")
print(get_time(annoy_np))
print("faiss_np")
print(get_time(faiss_np))
print("faiss_t_np")
print(get_time(faiss_t_np))
print("faiss_t_t")
print(get_time(faiss_t_t))
print("faiss_gpu")
print(get_time(faiss_gpu))

#import pdb; pdb.set_trace()
