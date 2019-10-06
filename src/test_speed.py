
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
import timeit

# cpu test
def annoy_cpu():
    r = reps_np[left_np, right_np]
    labels, distances = annoy_index.topk(r, 8)
    for le, ri, l, d in zip(): 

## numpy



## torch



# gpu test
# just torch
