"""
Visualization methods.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
import sklearn.decomposition as skd
import sklearn.manifold as skm

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

from fiftyone.brain.visualization import (
    VisualizationResults,
    UMAPVisualizationConfig,
    TSNEVisualizationConfig,
    PCAVisualizationConfig,
)
import fiftyone.brain.internal.models as fbm


logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "simple-resnet-cifar10"
_DEFAULT_BATCH_SIZE = 16


def compute_visualization(
    samples,
    patches_field,
    embeddings,
    brain_key,
    num_dims,
    method,
    config,
    model,
    batch_size,
    force_square,
    alpha,
    **kwargs,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if model is None and embeddings is None:
        model = fbm.load_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = _parse_config(
        config, embeddings_field, patches_field, method, num_dims, **kwargs
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

    logger.info("Generating visualization...")
    points = brain_method.fit(embeddings)

    results = VisualizationResults(samples, points, config)
    brain_method.save_run_results(samples, brain_key, results)

    return results


class Visualization(fob.BrainMethod):
    """Base class for embedding visualization methods.

    Args:
        config: a :class:`fiftyone.brain.visualization.VisualizationConfig`
    """

    def fit(self, embeddings):
        """Computes visualization coordinates for the given embeddings.

        Args:
            embeddings: a ``num_samples x num_features`` array of embeddings

        Returns:
            a ``num_samples x num_dims`` array of coordinates
        """
        raise NotImplementedError("subclass must implement fit()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass


class UMAPVisualization(Visualization):
    """Uniform Manifold Approximation and Projection (UMAP) embedding
    visualization.

    Args:
        config: a :class:`fiftyone.brain.visualization.UMAPVisualizationConfig`
    """

    def fit(self, embeddings):
        _ensure_umap()

        import umap

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
    """t-distributed Stochastic Neighbor Embedding (t-SNE) visualization.

    Args:
        config: a :class:`fiftyone.brain.visualization.TSNEVisualizationConfig`
    """

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
    """Principal component analysis (PCA) embedding visualization.

    Args:
        config: a :class:`fiftyone.brain.visualization.PCAVisualizationConfig`
    """

    def fit(self, embeddings):
        _pca = skd.PCA(
            n_components=self.config.num_dims,
            svd_solver=self.config.svd_solver,
            random_state=self.config.seed,
        )
        return _pca.fit_transform(embeddings)


def _parse_config(
    config, embeddings_field, patches_field, method, num_dims, **kwargs
):
    if config is not None:
        return config

    if method is None or method == "umap":
        config_cls = UMAPVisualizationConfig
    elif method == "tsne":
        config_cls = TSNEVisualizationConfig
    elif method == "pca":
        config_cls = PCAVisualizationConfig
    else:
        raise ValueError("Unsupported method '%s'" % method)

    return config_cls(
        embeddings_field=embeddings_field,
        patches_field=patches_field,
        num_dims=num_dims,
        **kwargs,
    )


def _ensure_umap():
    try:
        etau.ensure_package("umap-learn")
    except:
        raise ImportError(
            "You must install the `umap-learn` package in order to use "
            "UMAP-based visualization. This is recommended, as UMAP is "
            "awesome! If you do not wish to install UMAP, try `method='tsne'` "
            "instead"
        )
