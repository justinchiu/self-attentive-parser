
import pickle

import numpy as np

from pathlib import Path
from typing import NamedTuple

import annoy
import faiss
import torch
import scatter

# copied from faiss/.../test_pytorch_faiss.py
# ...no idea why there isn't built in support for torch

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def update_chart(chart, labels, distances, left, right):
    for le, ri, l, d in zip(
        left, right,
        labels, distances,
    ):
        np.logaddexp.at(chart[le, ri], l, d)

def chart_torch(T, N, labels, distances, flat_indices, chart_weight):
    # only one index right now
    cells = scatter.scatter_lse(
        distances,
        labels,
        dim = -1,
        dim_size = N,
        fill_value = float("-inf"),
    )
    flat_chart = torch.zeros((T+1) * (T+1), N, device=distances.device)
    flat_chart.fill_(float("-inf"))
    flat_chart = flat_chart.scatter(
        0,
        flat_indices.unsqueeze(-1).expand_as(cells),
        cells,
    )
    mask = flat_chart > float("-inf")
    filler = chart_weight.to(flat_chart.device).expand_as(flat_chart).masked_fill(mask, float("-inf"))
    flat_chart = torch.stack((flat_chart, filler), -1).logsumexp(-1)
    return flat_chart.view(T+1, T+1, flat_chart.shape[-1])


def get_index_prefix(index_base_path, full_model_path, nn_prefix):
    # models/bert*.pt
    index_base_path = Path(index_base_path)
    full_model_path = Path(full_model_path)
    fname = index_base_path / full_model_path.stem / nn_prefix
    return fname


def get_index_paths(prefix, num_indices, library):
    suffix = ".annoy" if library == "annoy" else ".faiss"
    nn_suffix = ".ann" + suffix
    info_suffix = ".info" + suffix
    key_suffix = ".key" + suffix + ".npy"
    label_suffix = ".label" + suffix + ".npy"
    labelkey_suffix = ".label2key" + suffix + ".npy"

    return (
        prefix.with_suffix(nn_suffix),
        prefix.with_suffix(info_suffix),
        prefix.with_suffix(key_suffix),
        prefix.with_suffix(label_suffix),
        prefix.with_suffix(labelkey_suffix),
    )

    """
    if num_indices > 1:
        return (
            [prefix.with_suffix(f".{i}{nn_suffix}") for i in range(num_indices)],
            [prefix.with_suffix(f".{i}{info_suffix}") for i in range(num_indices)],
            [prefix.with_suffix(f".{i}{key_suffix}") for i in range(num_indices)],
        )
    else:
        return (
            [prefix.with_suffix(nn_suffix)],
            [prefix.with_suffix(info_suffix)],
            [prefix.with_suffix(key_suffix)],
        )
    """


# Constructor for annoy indices
def init_annoy(dim, metric, num_labels):
    metric = "euclidean" if metric == "l2" else metric
    return (
        # index
        annoy.AnnoyIndex(dim, metric=metric),
        # span info
        [],
        # label maps
        [[] for _ in range(num_labels)],
    )
    """
    return (
        # index
        [annoy.AnnoyIndex(dim, metric=metric) for _ in range(num_indices)],
        # span info
        [[] for _ in range(num_indices)],
    )
    """


# Constructor for faiss indices
def init_faiss(dim, metric, num_labels, pca=False, in_dim=2000):
    nlist = 100
    def make_index():
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        if pca:
            # No idea what eigen_power: float or random_rotation: bool arguments of PCAMatrix do
            pca_matrix = faiss.PCAMatrix(in_dim, dim)
            index = faiss.IndexPreTransform(pca_matrix, index)
        return index
    #constring = f"PCA{dim},IVF{nlist},Flat" if pca else f"IVF{nlist},Flat"
    constring = f"IVF4096,PQ32"
    #constring = f"IVF262144_HNSW32,PQ32"
    metric_map = {
        "dot": faiss.METRIC_INNER_PRODUCT,
        "l2": faiss.METRIC_L2
    }
    f_metric = metric_map[metric]
    return (
        # index
        faiss.index_factory(dim, constring, f_metric),
        # span_info
        [],
        # key
        [],
        # labels
        [],
        # label to key idxs
        [[] for _ in range(num_labels)],
    )

init_fns = {
    "faiss": init_faiss,
    "annoy": init_annoy,
}

class SpanInfo(NamedTuple):
    label_idx: int
    label: str
    sen_idx: int
    left: int
    right: int


class AnnoyIndex:
    def __init__(
        self,
        dim = 256,
        metric = "dot",
        num_indices = 1,
        prefix = None,
    ):
        self.dim = dim
        self.metric = metric
        self.num_indices = num_indices

        # dispatch to respective constructor
        self.raw_indices, self.raw_span_infos = init_annoy(
            dim, metric, num_indices)

        # filepath prefix for saving and loading
        self.prefix = prefix

    def add_item(self, key, value, index=0):
        # key is the vector
        # value is the info
        idx = len(self.raw_span_infos[index])
        self.raw_indices[index].add_item(idx, key)
        self.raw_span_infos[index].append(value)

    def add(self, keys, values, indices):
        for key, value, index in zip(keys, values, indices):
            self.add_item(key, value, index)

    def _topk(self, key, k, label_only=True):
        # only labels (instead of structs) for now
        idx_and_distances = [
            index.get_nns_by_vector(key, k, include_distances=True)
            for index in self.raw_indices
        ]
        """
        labels = np.array([
            [
                self.raw_span_infos[index][idx].label_idx
                for idx, dist in zip(*idx_and_distance)
            ] for index, idx_and_distance in enumerate(idx_and_distances)
        ])
        distances = np.array([
            [dist for idx, dist in zip(*idx_and_distance)]
            for idx_and_distance in idx_and_distances
        ])
        """
        labels = np.array([
            self.raw_span_infos[index][idx].label_idx
            for index, idx_and_distance in enumerate(idx_and_distances)
            for idx, dist in zip(*idx_and_distance)
        ])
        distances = np.array([
            dist
            for idx_and_distance in idx_and_distances
            for idx, dist in zip(*idx_and_distance)
        ])
        # only return label, return span_info later?
        return labels, distances

    def topk(self, keys, k, label_only=True):
        # doesn't work yet, untested
        labels, distances = list(zip(*[self._topk(key, k, label_only) for key in keys]))
        assert label_only
        labels = np.stack(labels)
        distances = np.stack(distances)
        #import pdb; pdb.set_trace()
        return labels, -distances if self.metric == "l2" else distances

    def build(self, n_trees=16):
        for t in self.raw_indices:
            t.build(n_trees)

    def load(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names, _ = get_index_paths(prefix, self.num_indices, "annoy")
        for i, (t, ann_name, info_name) in enumerate(zip(
            self.raw_indices, ann_names, info_names,
        )):
            t.load(str(ann_name))
            self.raw_span_infos[i] = pickle.load(open(str(info_name), "rb"))

    def save(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names, _ = get_index_paths(prefix, self.num_indices, "annoy")
        for t, span_info, ann_name, info_name in zip(
            self.raw_indices, self.raw_span_infos, ann_names, info_names,
        ):
            ann_name.parent.mkdir(parents=True, exist_ok=True)
            t.save(str(ann_name))
            pickle.dump(span_info, open(info_name, "wb"))


class FaissIndex:
    def __init__(
        self,
        dim = 256,
        metric = "dot",
        num_labels = 1,
        prefix = None,
        pca = False,
    ):
        self.dim = dim
        self.metric = metric
        self.num_labels = num_labels

        # dispatch to respective constructor
        # NOTE: REMOVED LABEL INDEX
        (
            self.raw_index,
            self.raw_span_infos,
            self.keys,
            self.labels,
            self.label2keys,
        ) = init_faiss(dim, metric, num_labels)

        # filepath prefix for saving and loading
        self.prefix = prefix

        self.device = torch.device("cpu")

    def add_item(self, key, value):
        self.raw_index.add(key)
        self.raw_span_infos.append(value)
        self.keys.append(key)
        idx = len(self.keys)
        lbl = value.label_idx
        self.labels.append(lbl)
        self.label2keys[lbl].append(idx)

    def add(self, keys, values, index=0):
        # key is the vector
        # value is the info
        self.raw_span_infos.extend(values)
        # loop over keys and values to add to label2keys
        for key, value in zip(keys, values):
            self.keys.append(key)
            idx = len(self.keys)
            lbl = value.label_idx
            self.labels.append(lbl)
            self.label2keys[lbl].append(idx)

    def build(self):
        self.keys = np.stack(self.keys)

        self.raw_index.train(self.keys)
        self.raw_index.add(self.keys)

        self.labels = np.array(self.labels)
        self.label2keys = [np.array(xs) for xs in self.label2keys]

    # can mess with this later
    def topk(self, keys, k, label_only=True):
        # only labels for now
        # TODO: fix this! flatten along indices and batch all queries
        D, I = self.raw_index.search(keys, k)
        labels = self.labels[I]
        nn_keys = self.keys[I]
        return labels, -D if self.metric == "l2" else D, nn_keys

    def topk_torch(self, keys, k, label_only=True):
        D, I = search_index_pytorch(self.raw_index, keys, k)
        I_np = I.detach().cpu().numpy()
        labels = torch.from_numpy(self.labels[I_np]).to(keys.device)
        nn_keys = keys.new(self.keys[I_np])
        return labels, -D if self.metric == "l2" else D, nn_keys

    def topk_torch_np(self, keys, k, label_only=True):
        D, I = search_index_pytorch(self.raw_index, keys, k)
        D = D.detach().cpu().numpy()
        I_np = I.detach().cpu().numpy()
        labels = self.labels[I_np]
        nn_keys = self.keys[I_np]
        return labels, -D if self.metric == "l2" else D, nn_keys

    def load(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_name, info_name, key_name, label_name, labelkey_name = get_index_paths(prefix, self.num_labels, "faiss")
        self.raw_index = faiss.read_index(str(ann_name))
        self.raw_span_infos = pickle.load(open(str(info_name), "rb"))
        self.keys = np.load(key_name)
        self.labels = np.load(label_name)
        self.label2keys = np.load(labelkey_name, allow_pickle=True)

    def save(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_name, info_name, key_name, label_name, labelkey_name = get_index_paths(prefix, self.num_labels, "faiss")
        ann_name.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.raw_index, str(ann_name))
        pickle.dump(self.raw_span_infos, open(info_name, "wb"))
        np.save(key_name, self.keys)
        np.save(label_name, self.labels)
        np.save(labelkey_name, self.label2keys)

    def to(self, device):
        if device == self.device:
            pass
        elif device == torch.device("cpu") or device < 0:
            # To cpu
            self.raw_index = faiss.index_gpu_to_cpu(self.raw_index)
        else:
            # To gpu
            if isinstance(device, torch.device):
                device = device.index
            res = faiss.StandardGpuResources()
            self.raw_index = faiss.index_cpu_to_gpu(res, device, self.raw_index)


def get_span_reps_infos(parser, treebank, batch_size=32):
    span_reps = []
    span_infos = []
    for start_index in range(0, len(treebank), batch_size):
        subbatch_trees = treebank[start_index:start_index+batch_size]
        subbatch_sentences = [
            [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            for tree in subbatch_trees
        ]
        span_representations = parser.parse_batch(
            subbatch_sentences,
            return_span_representations = True,
        )
        for sub_index, (tree, chart) in enumerate(zip(subbatch_trees, span_representations)):
            train_index = start_index + sub_index
            chart = chart.cpu().numpy()
            # tree.leaves(): T
            # chart: T+1 x T+1 x H (span reps)
            T = len(list(tree.leaves()))
            parse = tree.convert()
            for length in range(1, T+1):
                for left in range(0, T+1-length):
                    right = left + length
                    label = parse.oracle_label(left, right)
                    # NOTE: removed ignore_empty
                    label_idx = parser.label_vocab.index(label)
                    span_rep = chart[left, right]
                    span_info = SpanInfo(
                        label_idx = label_idx,
                        label = label,
                        sen_idx = train_index,
                        left = left,
                        right = right,
                    )
                    span_reps.append(span_rep)
                    span_infos.append(span_info)
    return span_reps, span_infos

