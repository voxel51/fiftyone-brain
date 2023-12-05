"""
Visualization methods.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
import sklearn.decomposition as skd
import sklearn.manifold as skm

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.expressions as foe
import fiftyone.core.plots as fop
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

from fiftyone.brain.visualization import (
    VisualizationResults,
    UMAPVisualizationConfig,
    TSNEVisualizationConfig,
    PCAVisualizationConfig,
    ManualVisualizationConfig,
)
import fiftyone.brain.internal.core.utils as fbu

umap = fou.lazy_import("umap")


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mobilenet-v2-imagenet-torch"
_DEFAULT_BATCH_SIZE = None


def compute_visualization(
    samples,
    patches_field,
    embeddings,
    points,
    brain_key,
    num_dims,
    method,
    model,
    force_square,
    alpha,
    batch_size,
    num_workers,
    skip_failures,
    **kwargs,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if method == "manual" and points is None:
        raise ValueError(
            "You must provide your own `points` when `method='manual'`"
        )

    if points is not None:
        method = "manual"
        model = None
        embeddings = None
        embeddings_field = None
        num_dims = _get_dimension(points)

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings,
            patches_field=patches_field,
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if (
        model is None
        and points is None
        and embeddings is None
        and not embeddings_exist
    ):
        model = _DEFAULT_MODEL
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = _parse_config(
        embeddings_field, model, patches_field, method, num_dims, **kwargs
    )

    brain_method = config.build()
    brain_method.ensure_requirements()

    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    if points is None:
        embeddings, sample_ids, label_ids = fbu.get_embeddings(
            samples,
            model=model,
            patches_field=patches_field,
            embeddings_field=embeddings_field,
            embeddings=embeddings,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
        )

        logger.info("Generating visualization...")
        points = brain_method.fit(embeddings)
    else:
        points, sample_ids, label_ids = fbu.parse_data(
            samples,
            patches_field=patches_field,
            data=points,
            data_type="points",
        )

    results = VisualizationResults(
        samples,
        config,
        brain_key,
        points,
        sample_ids=sample_ids,
        label_ids=label_ids,
    )

    brain_method.save_run_results(samples, brain_key, results)

    return results


def values(results, path_or_expr):
    samples = results.view
    patches_field = results.config.patches_field
    if patches_field is not None:
        ids = results.current_label_ids
    else:
        ids = results.current_sample_ids

    return fbu.get_values(
        samples, path_or_expr, ids, patches_field=patches_field
    )


def visualize(
    results,
    labels=None,
    sizes=None,
    classes=None,
    backend="plotly",
    **kwargs,
):
    points = results.current_points
    samples = results.view
    patches_field = results.config.patches_field
    good_inds = results._curr_good_inds
    if patches_field is not None:
        ids = results.current_label_ids
    else:
        ids = results.current_sample_ids

    if good_inds is not None:
        if etau.is_container(labels) and not _is_expr(labels):
            labels = fbu.filter_values(
                labels, good_inds, patches_field=patches_field
            )

        if etau.is_container(sizes) and not _is_expr(sizes):
            sizes = fbu.filter_values(
                sizes, good_inds, patches_field=patches_field
            )

    if labels is not None and _is_expr(labels):
        labels = fbu.get_values(
            samples, labels, ids, patches_field=patches_field
        )

    if sizes is not None and _is_expr(sizes):
        sizes = fbu.get_values(
            samples, sizes, ids, patches_field=patches_field
        )

    return fop.scatterplot(
        points,
        samples=samples,
        ids=ids,
        link_field=patches_field,
        labels=labels,
        sizes=sizes,
        classes=classes,
        backend=backend,
        **kwargs,
    )


def _is_expr(arg):
    return isinstance(arg, (foe.ViewExpression, dict))


class Visualization(fob.BrainMethod):
    def fit(self, embeddings):
        raise NotImplementedError("subclass must implement fit()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields


class UMAPVisualization(Visualization):
    def ensure_requirements(self):
        fou.ensure_package(
            "umap-learn>=0.5",
            error_msg=(
                "You must install the `umap-learn>=0.5` package in order to "
                "use UMAP-based visualization. This is recommended, as UMAP "
                "is awesome! If you do not wish to install UMAP, try "
                "`method='tsne'` instead"
            ),
        )

    def fit(self, embeddings):
        _umap = umap.UMAP(
            n_components=self.config.num_dims,
            n_neighbors=self.config.num_neighbors,
            metric=self.config.metric,
            min_dist=self.config.min_dist,
            random_state=self.config.seed,
            verbose=self.config.verbose,
        )
        return _umap.fit_transform(embeddings)


class TSNEVisualization(Visualization):
    def fit(self, embeddings):
        if self.config.pca_dims is not None:
            _pca = skd.PCA(
                n_components=self.config.pca_dims,
                svd_solver=self.config.svd_solver,
                random_state=self.config.seed,
            )
            embeddings = _pca.fit_transform(embeddings)

        embeddings = embeddings.astype(np.float32, copy=False)

        verbose = 2 if self.config.verbose else 0

        _tsne = skm.TSNE(
            n_components=self.config.num_dims,
            perplexity=self.config.perplexity,
            learning_rate=self.config.learning_rate,
            metric=self.config.metric,
            init="pca",  # "random" or "pca"
            n_iter=self.config.max_iters,
            random_state=self.config.seed,
            verbose=verbose,
        )
        return _tsne.fit_transform(embeddings)


class PCAVisualization(Visualization):
    def fit(self, embeddings):
        _pca = skd.PCA(
            n_components=self.config.num_dims,
            svd_solver=self.config.svd_solver,
            random_state=self.config.seed,
        )
        return _pca.fit_transform(embeddings)


class ManualVisualization(Visualization):
    def fit(self, embeddings):
        raise NotImplementedError(
            "The low-dimensional representation must be manually provided "
            "when using this method"
        )


def _parse_config(
    embeddings_field, model, patches_field, method, num_dims, **kwargs
):
    if method is None:
        method = "umap"

    if method == "umap":
        config_cls = UMAPVisualizationConfig
    elif method == "tsne":
        config_cls = TSNEVisualizationConfig
    elif method == "pca":
        config_cls = PCAVisualizationConfig
    elif method == "manual":
        config_cls = ManualVisualizationConfig
    else:
        raise ValueError("Unsupported method '%s'" % method)

    return config_cls(
        embeddings_field=embeddings_field,
        model=model,
        patches_field=patches_field,
        num_dims=num_dims,
        **kwargs,
    )


def _get_dimension(points):
    if isinstance(points, dict):
        points = next(iter(points.values()), None)

    if isinstance(points, list):
        points = next(iter(points), None)

    if points is None:
        return 2

    return points.shape[-1]
