"""
Duplicates methods.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import itertools
import logging
import multiprocessing

import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

import eta.core.utils as etau

import fiftyone.core.aggregations as foa
import fiftyone.core.brain as fob
import fiftyone.core.context as foc
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.patches as fop
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

from fiftyone.brain.duplicates import DuplicatesConfig, DuplicatesResults
import fiftyone.brain.internal.models as fbm


logger = logging.getLogger(__name__)


_ALLOWED_PATCH_FIELD_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.Polyline,
    fol.Polylines,
)

_DEFAULT_MODEL = "simple-resnet-cifar10"
_DEFAULT_BATCH_SIZE = 16

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory
_COSINE_HACK_ATTR = "_cosine_hack"


def compute_exact_duplicates(samples, num_workers, skip_failures):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if num_workers is None:
        if samples.media_type == fom.VIDEO:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = 1

    logger.info("Computing filehashes...")

    method = "md5" if samples.media_type == fom.VIDEO else None

    if num_workers == 1:
        hashes = _compute_filehashes(samples, method)
    else:
        hashes = _compute_filehashes_multi(samples, method, num_workers)

    num_missing = sum(h is None for h in hashes)
    if num_missing > 0:
        msg = "Failed to compute %d filehashes" % num_missing
        if skip_failures:
            logger.warning(msg)
        else:
            raise ValueError(msg)

    dup_ids = []
    observed_hashes = set()
    for _id, _hash in hashes.items():
        if _hash is None:
            continue

        if _hash in observed_hashes:
            dup_ids.append(_id)
        else:
            observed_hashes.add(_hash)

    return dup_ids


def compute_duplicates(
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

    if patches_field is not None:
        fov.validate_collection_label_fields(
            samples, patches_field, _ALLOWED_PATCH_FIELD_TYPES
        )

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = DuplicatesConfig(
        embeddings_field=embeddings_field,
        model=model,
        patches_field=patches_field,
        metric=metric,
    )
    brain_method = config.build()
    brain_method.register_run(samples, brain_key)

    if model is not None or (embeddings is None and embeddings_field is None):
        if etau.is_str(model):
            model = foz.load_zoo_model(model)
        elif model is None:
            model = fbm.load_model(_DEFAULT_MODEL)
            if batch_size is None:
                batch_size = _DEFAULT_BATCH_SIZE

        logger.info("Generating embeddings...")

        if patches_field is None:
            embeddings = samples.compute_embeddings(
                model, batch_size=batch_size
            )
        else:
            embeddings = samples.compute_patch_embeddings(
                model,
                patches_field,
                handle_missing="skip",
                batch_size=batch_size,
                force_square=force_square,
                alpha=alpha,
            )

    if embeddings_field is not None:
        embeddings = samples.values(embeddings_field)

    if isinstance(embeddings, dict):
        _embeddings = []
        for _id in samples.values("id"):
            e = embeddings.get(_id, None)
            if e is not None:
                _embeddings.append(e)

        embeddings = np.concatenate(_embeddings, axis=0)

    results = DuplicatesResults(samples, embeddings, config)

    logger.info("Generating index...")

    neighbors = init_neighbors(embeddings, metric)
    results._neighbors = neighbors

    brain_method.save_run_results(samples, brain_key, results)

    logger.info("Index complete")

    return results


def init_neighbors(embeddings, metric):
    # Center embeddings
    embeddings = np.asarray(embeddings)
    embeddings -= embeddings.mean(axis=0, keepdims=True)

    # For small datasets, compute entire distance matrix
    num_embeddings = len(embeddings)
    if num_embeddings <= _MAX_PRECOMPUTE_DISTS:
        embeddings = skm.pairwise_distances(embeddings, metric=metric)
        metric = "precomputed"
    else:
        logger.info(
            "Computing neighbors for %d embeddings; this may take awhile...",
            num_embeddings,
        )

    #
    # For large datasets, use ``NearestNeighbors``
    #
    # @todo upper-bound number of allowed dimensions here? For many samples
    # with many dimensions, this will be impractically slow...
    #

    # Nearest neighbors does not directly support cosine distance, so we
    # approximate via euclidean distance on unit-norm embeddings
    if metric == "cosine":
        cosine_hack = True
        embeddings = skp.normalize(embeddings, axis=1)
        metric = "euclidean"
    else:
        cosine_hack = False

    neighbors = skn.NearestNeighbors(metric=metric)
    setattr(neighbors, _COSINE_HACK_ATTR, cosine_hack)

    neighbors.fit(embeddings)

    return neighbors


def find_duplicates(results, thresh, fraction):
    _ensure_neighbors(results)

    samples = results._samples
    neighbors = results._neighbors
    embeddings = results.embeddings
    metric = results.config.metric
    patches_field = results.config.patches_field

    num_embeddings = len(embeddings)

    logger.info("Computing duplicates...")

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

    if patches_field is not None:
        id_path = samples._get_label_field_path(patches_field, "id")
        ids = samples.values(id_path, unwind=True)
    else:
        ids = samples.values("id")

    #
    # Locate nearest non-duplicate for each duplicate
    #

    unique_inds = np.array(sorted(keep))
    dup_inds = np.array([i for i in range(num_embeddings) if i not in keep])

    if num_embeddings <= _MAX_PRECOMPUTE_DISTS:
        # Use pre-computed distances
        dists = neighbors._fit_X[unique_inds, :][:, dup_inds]
        min_inds = np.argmin(dists, axis=0)
    else:
        neighbors = skn.NearestNeighbors(metric=metric)
        neighbors.fit(embeddings[unique_inds, :])
        min_inds = neighbors.kneighbors(
            embeddings[dup_inds, :], n_neighbors=1, return_distance=False
        )

    unique_ids = np.array([_id for idx, _id in enumerate(ids) if idx in keep])
    dup_ids = np.array([_id for idx, _id in enumerate(ids) if idx not in keep])
    nearest_ids = np.array([ids[unique_inds[i]] for i in min_inds])

    results.thresh = thresh
    results.unique_ids = unique_ids
    results.dup_ids = dup_ids
    results.nearest_ids = nearest_ids

    logger.info("Duplicates computation complete")


def find_unique(results, count):
    _ensure_neighbors(results)

    samples = results._samples
    neighbors = results._neighbors
    embeddings = results.embeddings
    patches_field = results.config.patches_field

    logger.info("Computing uniques...")

    num_keep = count
    num_embeddings = len(embeddings)

    keep, thresh = _remove_duplicates_count(
        neighbors, num_keep, num_embeddings
    )

    if patches_field is not None:
        id_path = samples._get_label_field_path(patches_field, "id")
        ids = samples.values(id_path, unwind=True)
    else:
        ids = samples.values("id")

    unique_ids = np.array([_id for idx, _id in enumerate(ids) if idx in keep])
    dup_ids = np.array([_id for idx, _id in enumerate(ids) if idx not in keep])

    results.thresh = thresh
    results.unique_ids = unique_ids
    results.dup_ids = dup_ids
    results.nearest_ids = None

    logger.info("Unique computation complete")


def plot_distances(results, bins, log, backend, **kwargs):
    _ensure_neighbors(results)

    dists, _ = results._neighbors.kneighbors(n_neighbors=1)
    metric = results.config.metric
    thresh = results.thresh

    if backend == "matplotlib":
        return _plot_distances_mpl(dists, metric, thresh, bins, log, **kwargs)

    return _plot_distances_plotly(dists, metric, thresh, bins, log, **kwargs)


def duplicates_view(results, field):
    samples = results._samples
    patches_field = results.config.patches_field
    dup_ids = results.dup_ids
    nearest_ids = results.nearest_ids

    if patches_field is not None and not isinstance(samples, fop.PatchesView):
        samples = samples.to_patches(patches_field)

    dups_map = defaultdict(list)
    for dup_id, nearest_id in zip(dup_ids, nearest_ids):
        dups_map[nearest_id].append(dup_id)

    ids = []
    labels = []
    for _id in samples.values("id"):
        _dup_ids = dups_map.get(_id, None)
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
    samples = results._samples
    patches_field = results.config.patches_field
    unique_ids = results.unique_ids

    if patches_field is not None and not isinstance(samples, fop.PatchesView):
        samples = samples.to_patches(patches_field)

    return samples.select(list(unique_ids))


def visualize_duplicates(results, viz_results, backend, **kwargs):
    _ensure_visualization(results, viz_results)

    samples = results._samples
    visualization = results._visualization
    dup_ids = results.dup_ids
    nearest_ids = results.nearest_ids

    ids = samples.values("id")
    inds_map = {_id: idx for idx, _id in enumerate(ids)}

    dup_inds = np.array([inds_map[_id] for _id in dup_ids])
    nearest_inds = np.array([inds_map[_id] for _id in nearest_ids])

    _dup_ids = set(dup_ids)
    _nearest_ids = set(nearest_ids)

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
        kwargs["edges"] = np.stack((dup_inds, nearest_inds), axis=1)
        kwargs["edges_title"] = "neighbors"

    return visualization.visualize(
        labels=labels,
        classes=["unique", "nearest", "duplicate"],
        backend=backend,
        **kwargs,
    )


def visualize_unique(results, viz_results, backend, **kwargs):
    _ensure_visualization(results, viz_results)

    samples = results._samples
    visualization = results._visualization
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

    return visualization.visualize(
        labels=labels, classes=["other", "unique"], backend=backend, **kwargs,
    )


class Duplicates(fob.BrainMethod):
    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass


def _ensure_neighbors(results):
    if results._neighbors is None:
        results._neighbors = init_neighbors(
            results.embeddings, results.config.metric
        )


def _ensure_visualization(results, viz_results):
    if viz_results is None:
        if results._visualization is None:
            viz_results = _generate_visualization(results)
        else:
            viz_results = results._visualization

    results._visualization = viz_results


def _generate_visualization(results):
    import fiftyone.brain as fb

    samples = results._samples
    embeddings = results.embeddings
    patches_field = results.config.patches_field

    if embeddings.shape[1] in {2, 3}:
        return fb.VisualizationResults(
            samples,
            embeddings,
            fb.VisualizationConfig(patches_field=patches_field, num_dims=2),
        )

    return fb.compute_visualization(
        samples, patches_field=patches_field, embeddings=embeddings, num_dims=2
    )


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
    import fiftyone.core.plots.plotly as fop

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

    figure.update_layout(**fop._DEFAULT_LAYOUT)
    figure.update_layout(**kwargs)

    if foc.is_jupyter_context():
        figure = fop.PlotlyNotebookPlot(figure)

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


def _compute_filehashes(samples, method):
    # ids, filepaths = samples.values(["id", "filepath"])
    ids, filepaths = samples.aggregate(
        [foa.Values("id"), foa.Values("filepath")]
    )

    with fou.ProgressBar(total=len(ids)) as pb:
        return {
            _id: _compute_filehash(filepath, method)
            for _id, filepath in pb(zip(ids, filepaths))
        }


def _compute_filehashes_multi(samples, method, num_workers):
    # ids, filepaths = samples.values(["id", "filepath"])
    ids, filepaths = samples.aggregate(
        [foa.Values("id"), foa.Values("filepath")]
    )

    methods = itertools.repeat(method)

    inputs = list(zip(ids, filepaths, methods))

    with fou.ProgressBar(total=len(inputs)) as pb:
        with multiprocessing.Pool(processes=num_workers) as pool:
            return {
                k: v
                for k, v in pb(
                    pool.imap_unordered(_do_compute_filehash, inputs)
                )
            }


def _compute_filehash(filepath, method):
    try:
        filehash = fou.compute_filehash(filepath)
        # filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return filehash


def _do_compute_filehash(args):
    _id, filepath, method = args
    try:
        filehash = fou.compute_filehash(filepath)
        # filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return _id, filehash
