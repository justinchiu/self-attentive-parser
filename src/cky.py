import numpy as np

import torch
import torch_struct as ts


def nk2ts(chart):
    # chart: batch x time x time x num_classes
    # need to make indices inclusive to be compatible with torchstruct
    # and add 1st dimension corresponding to size of semiring
    return chart[:,:-1, 1:]

def batch_marg(chart, semiring=ts.MaxSemiring, lengths=None):
    chart_ts = nk2ts(chart).clone()
    chart_ts[:,0,lengths-1,0].fill_(-1e8) 
    model = ts.CKY_CRF
    struct = model(semiring)
    return struct.marginals(chart_ts, lengths=lengths-1)

def exclusive_spans(spans):
    # change endpoints of spans: batch x n x n x *
    # to be exclusive
    spans[:,2] += 1
    return spans

def pad(x, batch_idxs):
    len_padded = batch_idxs.max_len
    bsz = batch_idxs.batch_size
    lens = batch_idxs.seq_lens_np
    H = x.shape[-1]

    # filter out sentence boundaries
    xmask = np.zeros(x.shape[0])
    start = 0
    for l in lens:
        xmask[start:start+l-1] = True
        start += l
    xmask = torch.BoolTensor(xmask).to(x.device)
    x = x[xmask]

    padded = x.new(
        bsz, len_padded, H,
        device = x.device,
    )
    padded.fill_(0)
    #mask = np.zeros(batch_idxs.seq_lens_np.sum())
    mask = np.zeros(bsz * len_padded)
    for i, l in enumerate(lens):
        mask[i * len_padded: i * len_padded + l - 1] = 1
    mask = torch.BoolTensor(mask).to(x.device)
    index = torch.arange(0, mask.shape[0], device=x.device)[mask]
    #"""
    padded = (padded
        .view(-1, H)
        .index_copy(
            0,
            index,
            x,
        )
    )
    return padded.view(bsz, len_padded, H)
