
import pickle

import numpy as np

from pathlib import Path
from typing import NamedTuple

import annoy
import faiss
import torch

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
        labels[0], distances[0],
    ):
        np.logaddexp.at(chart[le, ri], l, d)

def update_chart_torch(chart, labels, distances, left, right):
    # only one index right now
    cells = scatter.scatter_lse(
        distances[0],
        labels[0],
        dim = -1,
        dim_size = len(self.label_vocab.values),
        fill_value = 0,
    )
    flat_chart = torch.zeros((T+1) * (T+1), len(self.label_vocab.values))
    flat_chart = flat_chart.scatter(
        0,
        flat_indices.unsqueeze(-1).expand_as(cells),
        cells,
    )
    chart = (flat_chart
        .view(T+1, T+1, flat_chart.shape[-1])
        .cpu()
        .numpy()
    )


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

    if num_indices > 1:
        return (
            [prefix.with_suffix(f".{i}{nn_suffix}") for i in range(num_indices)],
            [prefix.with_suffix(f".{i}{info_suffix}") for i in range(num_indices)],
        )
    else:
        return [prefix.with_suffix(nn_suffix)], [prefix.with_suffix(info_suffix)]


# Constructor for annoy indices
def init_annoy(dim, metric, num_indices):
    metric = "euclidean" if metric == "l2" else metric
    return (
        # index
        [annoy.AnnoyIndex(dim, metric=metric) for _ in range(num_indices)],
        # span info
        [[] for _ in range(num_indices)],
    )


# Constructor for faiss indices
def init_faiss(dim, metric, num_indices, pca=False, in_dim=2000):
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
        [faiss.index_factory(dim, constring, f_metric) for _ in range(num_indices)],
        #[make_index() for _ in range(num_indices)],
        [[] for _ in range(num_indices)],
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
        # only return label, return span_info later?
        return labels, distances

    def topk(self, keys, k, label_only=True):
        # doesn't work yet, untested
        labels, distances = list(zip(*[self._topk(key, k, label_only) for key in keys]))
        assert label_only
        labels = np.stack(labels, 1)
        distances = np.stack(distances, 1)
        return labels, distances

    def build(self, n_trees=16):
        for t in self.raw_indices:
            t.build(n_trees)

    def load(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices, "annoy")
        for i, (t, ann_name, info_name) in enumerate(zip(
            self.raw_indices, ann_names, info_names,
        )):
            t.load(str(ann_name))
            self.raw_span_infos[i] = pickle.load(open(str(info_name), "rb"))

    def save(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices, "annoy")
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
        num_indices = 1,
        prefix = None,
        pca = False,
    ):
        self.dim = dim
        self.metric = metric
        self.num_indices = num_indices

        # dispatch to respective constructor
        self.raw_indices, self.raw_span_infos = init_faiss(
            dim, metric, num_indices)

        # filepath prefix for saving and loading
        self.prefix = prefix

    def add_item(self, key, value, index=0):
        self.raw_indices[index].add(key)
        self.raw_span_infos[index].append(value)

    def add(self, keys, values, index=0):
        # key is the vector
        # value is the info
        self.raw_indices[index].add(keys)
        self.raw_span_infos[index].extend(values)

    # can mess with this later
    def topk(self, keys, k, label_only=True):
        # only labels for now
        distance_and_idxs = [
            index.search(keys, k)
            for index in self.raw_indices
        ]
        # distance_and_idxs: [
        #   [
        #       (
        #           # distance
        #           nq x k, float32
        #           # idx
        #           nq x k, int64
        #       )
        #   ]
        #   for each index
        # ]
        #labels = torch.LongTensor([
        labels = np.array([
            [
                [
                    self.raw_span_infos[index][idx].label_idx
                    for idx in idxs
                ] for idxs in idxss#.tolist()
            ] for index, (distss, idxss)in enumerate(distance_and_idxs)
        ], dtype=np.int32)
        #distances = torch.Tensor([
        distances = np.array([
            distss for distss, _ in distance_and_idxs
        ])
        # only return label, return span_info later?
        return labels, distances

    def topk_torch(self, keys, k, label_only=True):
        distance_and_idxs = [
            search_index_pytorch(index, keys, k)
            for index in self.raw_indices
        ]
        # distance_and_idxs: [
        #   [
        #       (
        #           # distance
        #           nq x k, float32
        #           # idx
        #           nq x k, int64
        #       )
        #   ]
        #   for each index
        # ]
        #import pdb; pdb.set_trace()
        labels = torch.LongTensor([
            [
                [
                    self.raw_span_infos[index][idx].label_idx
                    for idx in idxs
                ] for idxs in idxss.tolist()
            ] for index, (distss, idxss)in enumerate(distance_and_idxs)
        ]).to(keys.device)
        distances = torch.stack([
            distss for distss, _ in distance_and_idxs
        ], 0)
        # only return label, return span_info later?
        return labels, distances

    def topk_torch_np(self, keys, k, label_only=True):
        distance_and_idxs = [
            [x.cpu().numpy() for x in search_index_pytorch(index, keys, k)]
            for index in self.raw_indices
        ]
        # distance_and_idxs: [
        #   [
        #       (
        #           # distance
        #           nq x k, float32
        #           # idx
        #           nq x k, int64
        #       )
        #   ]
        #   for each index
        # ]
        labels = np.array([
            [
                [
                    self.raw_span_infos[index][idx].label_idx
                    for idx in idxs
                ] for idxs in idxss#.tolist()
            ] for index, (distss, idxss)in enumerate(distance_and_idxs)
        ], dtype=np.int32)
        distances = np.array([
            distss for distss, _ in distance_and_idxs
        ])
        # only return label, return span_info later?
        return labels, distances


    def build(self):
        pass

    def load(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices, "faiss")
        for i, (t, ann_name, info_name) in enumerate(zip(
            self.raw_indices, ann_names, info_names,
        )):
            self.raw_indices[i] = faiss.read_index(str(ann_name))
            self.raw_span_infos[i] = pickle.load(open(str(info_name), "rb"))

    def save(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices, "faiss")
        for t, span_info, ann_name, info_name in zip(
            self.raw_indices, self.raw_span_infos, ann_names, info_names,
        ):
            ann_name.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(t, str(ann_name))
            pickle.dump(span_info, open(info_name, "wb"))

    def to(self, device):
        if isinstance(device, torch.device):
            device = device.index
        res = faiss.StandardGpuResources()
        self.raw_indices = [
            faiss.index_cpu_to_gpu(res, device, x)
            for x in self.raw_indices
        ]
