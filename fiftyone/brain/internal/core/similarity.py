"""
Similarity methods.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import logging

from bson import ObjectId
import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

import eta.core.utils as etau

from fiftyone import ViewField as F
import fiftyone.core.brain as fob
import fiftyone.core.context as foc
import fiftyone.core.fields as fof
import fiftyone.core.patches as fop
import fiftyone.core.stages as fos
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

from fiftyone.brain.similarity import SimilarityConfig, SimilarityResults


logger = logging.getLogger(__name__)

_AGGREGATIONS = {"mean": np.mean, "min": np.min, "max": np.max}

_DEFAULT_MODEL = "mobilenet-v2-imagenet-torch"
_DEFAULT_BATCH_SIZE = None

# _DEFAULT_MODEL = "simple-resnet-cifar10"
# _DEFAULT_BATCH_SIZE = 16

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory
_COSINE_HACK_ATTR = "_cosine_hack"


def compute_similarity(
    samples,
    patches_field,
    embeddings,
    brain_key,
    metric,
    model,
    batch_size,
    force_square,
    alpha,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if model is None and embeddings is None:
        model = foz.load_zoo_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = SimilarityConfig(
        embeddings_field=embeddings_field,
        model=model,
        patches_field=patches_field,
        metric=metric,
    )
    brain_method = config.build()
    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    if model is not None:
        if etau.is_str(model):
            model = foz.load_zoo_model(model)

        if patches_field is not None:
            logger.info("Computing patch embeddings...")
            embeddings = samples.compute_patch_embeddings(
                model,
                patches_field,
                embeddings_field=embeddings_field,
                handle_missing="skip",
                batch_size=batch_size,
                force_square=force_square,
                alpha=alpha,
            )
        else:
            logger.info("Computing embeddings...")
            embeddings = samples.compute_embeddings(
                model,
                embeddings_field=embeddings_field,
                batch_size=batch_size,
            )

    if embeddings_field is not None:
        embeddings = samples.values(embeddings_field)
        embeddings = [e for e in embeddings if e is not None and e.size > 0]
        if patches_field is not None:
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            embeddings = np.stack(embeddings)

    if isinstance(embeddings, dict):
        embeddings = [embeddings[_id] for _id in samples.values("id")]
        embeddings = [e for e in embeddings if e is not None and e.size > 0]
        embeddings = np.concatenate(embeddings, axis=0)

    results = SimilarityResults(samples, embeddings, config)
    brain_method.save_run_results(samples, brain_key, results)

    return results


class Similarity(fob.BrainMethod):
    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass


class NeighborsHelper(object):
    def __init__(self, embeddings, metric):
        self.embeddings = embeddings
        self.metric = metric

        self._initialized = False
        self._full_dists = None
        self._curr_keep_inds = None
        self._curr_dists = None
        self._curr_neighbors = None

    def get_distances(self, keep_inds=None):
        self._init()

        if self._same_keep_inds(keep_inds):
            return self._curr_dists

        if keep_inds is not None:
            dists, _ = self._build(keep_inds=keep_inds, build_neighbors=False)
        else:
            dists = self._full_dists

        self._curr_keep_inds = keep_inds
        self._curr_dists = dists
        self._curr_neighbors = None

        return dists

    def get_neighbors(self, keep_inds=None):
        self._init()

        if self._curr_neighbors is not None and self._same_keep_inds(
            keep_inds
        ):
            return self._curr_neighbors, self._curr_dists

        dists, neighbors = self._build(keep_inds=keep_inds)

        self._curr_keep_inds = keep_inds
        self._curr_dists = dists
        self._curr_neighbors = neighbors

        return neighbors, dists

    def _same_keep_inds(self, keep_inds):
        if keep_inds is None and self._curr_keep_inds is None:
            return True

        if (
            isinstance(keep_inds, np.ndarray)
            and isinstance(self._curr_keep_inds, np.ndarray)
            and keep_inds.size == self._curr_keep_inds.size
            and (keep_inds == self._curr_keep_inds).all()
        ):
            return True

        return False

    def _init(self):
        if self._initialized:
            return

        # Pre-compute all pairwise distances if number of embeddings is small
        if len(self.embeddings) <= _MAX_PRECOMPUTE_DISTS:
            dists, _ = self._build_precomputed(
                self.embeddings, build_neighbors=False
            )
        else:
            dists = None

        self._initialized = True
        self._full_dists = dists
        self._curr_keep_inds = None
        self._curr_dists = dists
        self._curr_neighbors = None

    def _build(self, keep_inds=None, build_neighbors=True):
        # Use full distance matrix if available
        if self._full_dists is not None:
            if keep_inds is not None:
                dists = self._full_dists[keep_inds, :][:, keep_inds]
            else:
                dists = self._full_dists

            if build_neighbors:
                neighbors = skn.NearestNeighbors(metric="precomputed")
                neighbors.fit(dists)
            else:
                neighbors = None

            return dists, neighbors

        # Must build index
        embeddings = self.embeddings

        if keep_inds is not None:
            embeddings = embeddings[keep_inds]

        if len(embeddings) <= _MAX_PRECOMPUTE_DISTS:
            dists, neighbors = self._build_precomputed(
                embeddings, build_neighbors=build_neighbors
            )
        else:
            dists = None
            neighbors = self._build_graph(embeddings)

        return dists, neighbors

    def _build_precomputed(self, embeddings, build_neighbors=True):
        logger.info("Generating index...")

        # Center embeddings
        embeddings = np.asarray(embeddings)
        embeddings -= embeddings.mean(axis=0, keepdims=True)

        dists = skm.pairwise_distances(embeddings, metric=self.metric)

        if build_neighbors:
            neighbors = skn.NearestNeighbors(metric="precomputed")
            neighbors.fit(dists)
        else:
            neighbors = None

        logger.info("Index complete")

        return dists, neighbors

    def _build_graph(self, embeddings):
        logger.info(
            "Generating neighbors graph for %d embeddings; this may take "
            "awhile...",
            len(embeddings),
        )

        # Center embeddings
        embeddings = np.asarray(embeddings)
        embeddings -= embeddings.mean(axis=0, keepdims=True)

        metric = self.metric

        if metric == "cosine":
            # Nearest neighbors does not directly support cosine distance, so
            # we approximate via euclidean distance on unit-norm embeddings
            cosine_hack = True
            embeddings = skp.normalize(embeddings, axis=1)
            metric = "euclidean"
        else:
            cosine_hack = False

        neighbors = skn.NearestNeighbors(metric=metric)
        neighbors.fit(embeddings)

        setattr(neighbors, _COSINE_HACK_ATTR, cosine_hack)

        logger.info("Index complete")

        return neighbors


def plot_distances(results, bins, log, backend, **kwargs):
    _ensure_neighbors(results)

    keep_inds = results._curr_keep_inds
    neighbors, _ = results._neighbors_helper.get_neighbors(keep_inds=keep_inds)
    metric = results.config.metric
    thresh = results.thresh

    dists, _ = neighbors.kneighbors(n_neighbors=1)

    if backend == "matplotlib":
        return _plot_distances_mpl(dists, metric, thresh, bins, log, **kwargs)

    return _plot_distances_plotly(dists, metric, thresh, bins, log, **kwargs)


def sort_by_similarity(results, query_ids, k, reverse, aggregation, mongo):
    _ensure_neighbors(results)

    samples = results.view
    sample_ids = results._curr_sample_ids
    label_ids = results._curr_label_ids
    keep_inds = results._curr_keep_inds
    patches_field = results.config.patches_field
    metric = results.config.metric

    if etau.is_str(query_ids):
        query_ids = [query_ids]

    if not query_ids:
        raise ValueError("At least one query ID must be provided")

    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            "Unsupported aggregation method '%s'. Supported values are %s"
            % (aggregation, tuple(_AGGREGATIONS.keys()))
        )

    #
    # Parse query (always using full index)
    #

    if patches_field is None:
        ids = results._sample_ids
    else:
        ids = results._label_ids

    bad_ids = []
    query_inds = []
    for query_id in query_ids:
        _inds = np.where(ids == query_id)[0]
        if _inds.size == 0:
            bad_ids.append(query_id)
        else:
            query_inds.append(_inds[0])

    if bad_ids:
        raise ValueError(
            "Query IDs %s were not included in this index" % bad_ids
        )

    #
    # Perform sorting
    #

    dists = results._neighbors_helper.get_distances()

    if dists is not None:
        if keep_inds is not None:
            dists = dists[keep_inds, :]

        dists = dists[:, query_inds]
    else:
        index_embeddings = results.embeddings
        if keep_inds is not None:
            index_embeddings = index_embeddings[keep_inds]

        query_embeddings = results.embeddings[query_inds]
        dists = skm.pairwise_distances(
            index_embeddings, query_embeddings, metric=metric
        )

    agg_fcn = _AGGREGATIONS[aggregation]
    dists = agg_fcn(dists, axis=1)

    inds = np.argsort(dists)
    if reverse:
        inds = np.flip(inds)

    if k is not None:
        inds = inds[:k]

    if patches_field is None:
        result_ids = list(sample_ids[inds])
    else:
        result_ids = list(label_ids[inds])

    #
    # Construct sorted view
    #

    stages = []

    if patches_field is None:
        stage = fos.Select(result_ids, ordered=True)
        stages.append(stage)
    else:
        result_sample_ids = _unique_no_sort(sample_ids[inds])
        stage = fos.Select(result_sample_ids, ordered=True)
        stages.append(stage)

        if k is not None:
            _ids = [ObjectId(_id) for _id in result_ids]
            stage = fos.FilterLabels(patches_field, F("_id").is_in(_ids))
            stages.append(stage)

    if mongo:
        pipeline = []
        for stage in stages:
            stage.validate(samples)
            pipeline.extend(stage.to_mongo(samples))

        return pipeline

    view = samples
    for stage in stages:
        view = view.add_stage(stage)

    return view


def find_duplicates(results, thresh, fraction):
    _ensure_neighbors(results)

    keep_inds = results._curr_keep_inds
    embeddings = results.embeddings
    metric = results.config.metric
    patches_field = results.config.patches_field

    neighbors, dists = results._neighbors_helper.get_neighbors(
        keep_inds=keep_inds
    )

    if keep_inds is not None:
        embeddings = embeddings[keep_inds]

    if patches_field is not None:
        ids = results._curr_label_ids
        logger.info("Computing duplicate patches...")
    else:
        ids = results._curr_sample_ids
        logger.info("Computing duplicate samples...")

    num_embeddings = len(embeddings)

    #
    # Detect duplicates
    #

    if fraction is not None:
        num_keep = int(round(min(max(0, 1.0 - fraction), 1) * num_embeddings))
        keep, thresh = _remove_duplicates_count(
            neighbors, num_keep, num_embeddings, init_thresh=thresh,
        )
    else:
        keep = _remove_duplicates_thresh(neighbors, thresh, num_embeddings)

    #
    # Locate nearest non-duplicate for each duplicate
    #

    unique_inds = sorted(keep)

    dup_inds = [idx for idx in range(num_embeddings) if idx not in keep]

    unique_inds = np.array(unique_inds)
    dup_inds = np.array(dup_inds)

    if dists is not None:
        # Use pre-computed distances
        min_inds = np.argmin(dists[unique_inds, :][:, dup_inds], axis=0)
    else:
        neighbors = skn.NearestNeighbors(metric=metric)
        neighbors.fit(embeddings[unique_inds, :])
        min_inds = neighbors.kneighbors(
            embeddings[dup_inds, :], n_neighbors=1, return_distance=False
        )

    unique_ids = [_id for idx, _id in enumerate(ids) if idx in keep]
    duplicate_ids = [_id for idx, _id in enumerate(ids) if idx not in keep]

    neighbors_map = defaultdict(list)
    for dup_id, min_ind in zip(duplicate_ids, min_inds):
        nearest_id = ids[unique_inds[min_ind]]
        neighbors_map[nearest_id].append(dup_id)

    results._thresh = thresh
    results._unique_ids = unique_ids
    results._duplicate_ids = duplicate_ids
    results._neighbors_map = dict(neighbors_map)

    logger.info("Duplicates computation complete")


def find_unique(results, count):
    _ensure_neighbors(results)

    keep_inds = results._curr_keep_inds
    neighbors, _ = results._neighbors_helper.get_neighbors(keep_inds=keep_inds)
    patches_field = results.config.patches_field
    num_embeddings = results.index_size

    if patches_field is not None:
        ids = results._curr_label_ids
        logger.info("Computing unique patches...")
    else:
        ids = results._curr_sample_ids
        logger.info("Computing unique samples...")

    # Find uniques
    keep, thresh = _remove_duplicates_count(neighbors, count, num_embeddings)

    unique_ids = [_id for idx, _id in enumerate(ids) if idx in keep]
    duplicate_ids = [_id for idx, _id in enumerate(ids) if idx not in keep]

    results._thresh = thresh
    results._unique_ids = unique_ids
    results._duplicate_ids = duplicate_ids
    results._neighbors_map = None

    logger.info("Uniqueness computation complete")


def duplicates_view(results, field):
    samples = results.view
    patches_field = results.config.patches_field
    neighbors_map = results.neighbors_map

    if patches_field is not None and not isinstance(samples, fop.PatchesView):
        samples = samples.to_patches(patches_field)

    ids = []
    labels = []
    for _id in samples.values("id"):
        _dup_ids = neighbors_map.get(_id, None)
        if _dup_ids is None:
            continue

        ids.append(_id)
        labels.append("nearest")

        ids.extend(_dup_ids)
        labels.extend(["duplicate"] * len(_dup_ids))

    dups_view = samples.select(ids, ordered=True)

    dups_view._dataset._add_sample_field_if_necessary(field, fof.StringField)
    dups_view.set_values(field, labels)

    return dups_view


def unique_view(results):
    samples = results.view
    patches_field = results.config.patches_field
    unique_ids = results.unique_ids

    if patches_field is not None and not isinstance(samples, fop.PatchesView):
        samples = samples.to_patches(patches_field)

    return samples.select(unique_ids)


def visualize_duplicates(results, visualization, backend, **kwargs):
    visualization = _ensure_visualization(results, visualization)

    samples = results.view
    duplicate_ids = results.duplicate_ids
    neighbors_map = results.neighbors_map

    ids = samples.values("id")

    _dup_ids = set(duplicate_ids)
    _nearest_ids = set(neighbors_map.keys())

    labels = []
    for _id in ids:
        if _id in _dup_ids:
            label = "duplicate"
        elif _id in _nearest_ids:
            label = "nearest"
        else:
            label = "unique"

        labels.append(label)

    # Only plotly backend supports drawing edges
    if backend == "plotly":
        kwargs["edges"] = _build_edges(ids, neighbors_map)
        kwargs["edges_title"] = "neighbors"

    with visualization.use_view(samples):
        return visualization.visualize(
            labels=labels,
            classes=["unique", "nearest", "duplicate"],
            backend=backend,
            **kwargs,
        )


def visualize_unique(results, visualization, backend, **kwargs):
    visualization = _ensure_visualization(results, visualization)

    samples = results.view
    unique_ids = results.unique_ids

    ids = samples.values("id")

    _unique_ids = set(unique_ids)

    labels = []
    for _id in ids:
        if _id in _unique_ids:
            label = "unique"
        else:
            label = "other"

        labels.append(label)

    with visualization.use_view(samples):
        return visualization.visualize(
            labels=labels,
            classes=["other", "unique"],
            backend=backend,
            **kwargs,
        )


def _unique_no_sort(values):
    seen = set()
    return [v for v in values if v not in seen and not seen.add(v)]


def _ensure_neighbors(results):
    if results._neighbors_helper is not None:
        return

    embeddings = results.embeddings
    metric = results.config.metric
    results._neighbors_helper = NeighborsHelper(embeddings, metric)


def _ensure_visualization(results, visualization):
    if visualization is not None:
        return visualization

    import fiftyone.brain as fb

    samples = results._samples
    embeddings = results.embeddings
    patches_field = results.config.patches_field

    if embeddings.shape[1] in {2, 3}:
        config = fb.VisualizationConfig(
            patches_field=patches_field, num_dims=2
        )
        return fb.VisualizationResults(samples, embeddings, config)

    return fb.compute_visualization(
        samples,
        patches_field=patches_field,
        embeddings=embeddings,
        num_dims=2,
        seed=51,
        verbose=True,
    )


def _build_edges(ids, neighbors_map):
    inds_map = {_id: idx for idx, _id in enumerate(ids)}

    edges = []
    for nearest_id, dup_ids in neighbors_map.items():
        nearest_ind = inds_map[nearest_id]
        for dup_id in dup_ids:
            dup_ind = inds_map[dup_id]
            edges.append((dup_ind, nearest_ind))

    return np.array(edges)


def _remove_duplicates_thresh(neighbors, thresh, num_embeddings):
    # When not using brute force, we approximate cosine distance by computing
    # Euclidean distance on unit-norm embeddings. ED = sqrt(2 * CD), so we need
    # to scale the threshold appropriately
    if getattr(neighbors, _COSINE_HACK_ATTR, False):
        thresh = np.sqrt(2.0 * thresh)

    inds = neighbors.radius_neighbors(radius=thresh, return_distance=False)

    keep = set(range(num_embeddings))
    for ind in range(num_embeddings):
        if ind in keep:
            keep -= {i for i in inds[ind] if i > ind}

    return keep


def _remove_duplicates_count(
    neighbors, num_keep, num_embeddings, init_thresh=None
):
    if init_thresh is not None:
        thresh = init_thresh
    else:
        thresh = 1

    thresh_lims = [0, None]
    num_target = num_keep
    num_keep = -1

    while True:
        keep = _remove_duplicates_thresh(neighbors, thresh, num_embeddings)
        num_keep_last = num_keep
        num_keep = len(keep)

        logger.info(
            "threshold: %f, kept: %d, target: %d",
            thresh,
            num_keep,
            num_target,
        )

        if num_keep == num_target or (
            num_keep == num_keep_last
            and thresh_lims[1] is not None
            and thresh_lims[1] - thresh_lims[0] < 1e-6
        ):
            break

        if num_keep < num_target:
            # Need to decrease threshold
            thresh_lims[1] = thresh
            thresh = 0.5 * (thresh_lims[0] + thresh)
        else:
            # Need to increase threshold
            thresh_lims[0] = thresh
            if thresh_lims[1] is not None:
                thresh = 0.5 * (thresh + thresh_lims[1])
            else:
                thresh *= 2

    return keep, thresh


def _plot_distances_plotly(dists, metric, thresh, bins, log, **kwargs):
    import plotly.graph_objects as go
    import fiftyone.core.plots.plotly as fopl

    counts, edges = np.histogram(dists, bins=bins)
    left_edges = edges[:-1]
    widths = edges[1:] - edges[:-1]
    customdata = np.stack((edges[:-1], edges[1:]), axis=1)

    hover_lines = [
        "<b>count: %{y}</b>",
        "distance: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]",
    ]
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    bar = go.Bar(
        x=left_edges,
        y=counts,
        width=widths,
        customdata=customdata,
        offset=0,
        marker_color="#FF6D04",
        hovertemplate=hovertemplate,
        showlegend=False,
    )

    traces = [bar]

    if thresh is not None:
        line = go.Scatter(
            x=[thresh, thresh],
            y=[0, max(counts)],
            mode="lines",
            line=dict(color="#17191C", width=3),
            hovertemplate="<b>thresh: %{x}</b><extra></extra>",
            showlegend=False,
        )
        traces.append(line)

    figure = go.Figure(traces)

    figure.update_layout(
        xaxis_title="nearest neighbor distance (%s)" % metric,
        yaxis_title="count",
        hovermode="x",
        yaxis_rangemode="tozero",
    )

    if log:
        figure.update_layout(yaxis_type="log")

    figure.update_layout(**fopl._DEFAULT_LAYOUT)
    figure.update_layout(**kwargs)

    if foc.is_jupyter_context():
        figure = fopl.PlotlyNotebookPlot(figure)

    return figure


def _plot_distances_mpl(
    dists, metric, thresh, bins, log, ax=None, figsize=None, **kwargs
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    counts, edges = np.histogram(dists, bins=bins)
    left_edges = edges[:-1]
    widths = edges[1:] - edges[:-1]

    ax.bar(
        left_edges,
        counts,
        width=widths,
        align="edge",
        color="#FF6D04",
        **kwargs,
    )

    if thresh is not None:
        ax.vlines(thresh, 0, max(counts), color="#17191C", linewidth=3)

    if log:
        ax.set_yscale("log")

    ax.set_xlabel("nearest neighbor distance (%s)" % metric)
    ax.set_ylabel("count")

    if figsize is not None:
        fig.set_size_inches(*figsize)

    plt.tight_layout()

    return fig
