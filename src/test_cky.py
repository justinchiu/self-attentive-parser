import torch
import torch_struct as ts

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_helper
import trees

N = 16
T = 5
num_labels = 3
chart = torch.randn(N, T+1, T+1, num_labels) * 5

chart_np = chart.cpu().numpy()
spans0 = []
for n in range(N):
    decoder_args = dict(
        sentence_len=T,
        label_scores_chart=chart_np[n],
        gold=None,
        label_vocab=None,
        is_train=False,
    )
    score, p_i, p_j, p_label, _ = chart_helper.decode(False, **decoder_args)
    spans0.append(sorted(list(zip(p_i, p_j, p_label))))

model = ts.CKY_CRF
max_struct = model(ts.MaxSemiring)

def gs(chart, struct):
    # don't allow root symbol to be empty
    chart[:,0,-1,0].fill_(-1e8)
    spans = struct.marginals(chart).nonzero()
    spans[:,2] += 1
    return spans

def cat(chart, dim):
    shape = list(chart.shape)
    shape[dim] = 1
    return torch.cat([ chart, torch.zeros(shape) ], dim=dim)

def cs(spans, spans0, n):
    return spans[spans.eq(n)[:,0]][:,1:].tolist(), spans0[n]

def meh(n):
    a, b = cs(spans, spans0, n)
    print(a)
    print(b)

spans = gs(chart[:,:-1, 1:], max_struct)

correct = [np.allclose(*cs(spans, spans0, n)) for n in range(N)]

print(all(correct))
import pdb; pdb.set_trace()
