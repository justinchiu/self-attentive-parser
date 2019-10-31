import pickle
import sys
import psutil

import torch
import numpy as np

import index

from pympler import muppy, summary
import gc

def print_mem():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    print(psutil.Process().memory_full_info().rss / 1e9)

prefix = index.get_index_prefix(
    index_base_path = "../index",
    full_model_path = "models/en_multibert_empty_l2_dev=94.94.pt",
    nn_prefix = "nl2-multi",
)
span_index = index.FaissIndex(num_labels = 112, metric="l2")
span_index.load(prefix)

shape = span_index.keys.shape

print("initialized index")
print_mem()
span_index.to(1)
print("to gpu")
print_mem()

span_index.reset()
print("reset index")
print_mem()

print("collecting garbage")
gc.collect()
print_mem()

new_keys = np.random.randn(*shape)
print("made new keys")
print_mem()

import pdb; pdb.set_trace()
