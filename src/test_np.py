import pickle
import sys

import torch
import numpy as np

import index

from pympler import muppy, summary
import gc

def print_mem():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

print ('NumPy version:', np.__version__, 'Python version:', sys.version)
f = np.zeros(0)
refcount = sys.getrefcount(f)
pickle.dumps(f)
print("Refcount delta:", sys.getrefcount(f) - refcount)

keys = np.load("../index/en_multibert_empty_l2_dev=94.94/nl2-multi.key.faiss.npy")

shape = keys.shape

print("initialized keys")
print_mem()

del keys
print("deleting keys")
print_mem()

print("collecting garbage")
gc.collect()
print_mem()

new_keys = np.empty(shape)
print("made new keys")
print_mem()
import pdb; pdb.set_trace()
