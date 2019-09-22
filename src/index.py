

from pathlib import Path
from typing import NamedTuple

from annoy import AnnoyIndex


def get_index_prefix(index_base_path, full_model_path, nn_prefix):
    nn_suffix = ".ann"
    info_suffix = ".info"

    # models/bert*.pt
    index_base_path = Path(index_base_path)
    full_model_path = Path(full_model_path)
    fname = index_base_path / full_model_path.stem / nn_prefix
    return fname

def get_index_paths(prefix, num_indices):
    if num_indices > 0:
        return (
            [fname.with_suffix(f".{i}{nn_suffix}") for i in range(num_indices)],
            [fname.with_suffix(f".{i}{info_suffix}") for i in range(num_indices)],
        )
    else:
        return [fname.with_suffix(nn_suffix)], [fname.with_suffix(info_suffix)]


# Constructor for annoy indices
def init_annoy(dim, metric, num_indices):
    return (
        # index
        [AnnoyIndex(dim, metric) for _ in range(num_indices)],
        # span info
        [[] for _ in range(num_indices)],
    )


class SpanInfo(NamedTuple):
    label_idx: int
    label: str
    sen_idx: int
    left: int
    right: int


# TODO: refactor out all nearest neighbour stuff into this class
class SpanIndex:
    def __init__(
        self,
        library = "annoy",
        dim = 250,
        metric = "dot",
        num_indices = 1,
        prefix = None,
    ):
        assert library in ["annoy"], "Unrecognized nearest neighbour library."
        self.library = library

        self.dim = dim
        self.metric = metric
        self.num_indices = num_indices

        # dispatch to respective constructor
        self.raw_indices, self.raw_span_infos = (
            init_annoy(dim, metric, num_indices)
            if library == "annoy"
            else None
        )

        # filepath prefix for saving and loading
        self.prefix = prefix

    def add_item(self, key, value, index=0):
        # key is the vector
        # value is the info
        idx = len(self.raw_span_infos[index])
        self.raw_indices[index].add_item(idx, key)
        self.raw_span_infos[index].append(value)

    def annoy_topk(self, key, k, label_only=True):
        # only labels for now
        idx_and_distance = [
            index.get_nns_by_vector(key, k, include_distances=True)
            for index in self.raw_indices
        ]
        labels = np.array([
            self.raw_span_infos[idx].label_idx
            for idx, dist in idx_and_distance
        ])
        distances = np.array([dist for idx, dist in idx_and_distance])
        # only return label
        return labels, distances

    def build(self, n_trees=16):
        for t in self.raw_indices:
            t.build(n_trees)

    def load(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices)
        for i, (t, ann_name, info_name) in enumerate(zip(
            self.raw_indices, ann_names, info_names,
        )):
            t.load(ann_name)
            self.raw_span_infos[i] = pickle.load(open(info_name, "rb"))

    def save(self, prefix=None):
        assert self.prefix or prefix, "One of self.prefix or prefix must not be None"
        prefix = self.prefix if prefix is None else prefix

        ann_names, info_names = get_index_paths(prefix, self.num_indices)
        for t, span_info, ann_name, info_name in zip(
            self.raw_indices, self.raw_span_infos, ann_names, info_names,
        ):
            ann_name.parent.mkdir(parents=True, exist_ok=True)
            t.build(16)
            t.save(str(ann_name))
            pickle.dump(span_info, open(info_name, "wb"))
