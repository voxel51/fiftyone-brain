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
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.utils.plot as foup
import fiftyone.zoo as foz

umap = fou.lazy_import(
    "umap", callback=lambda: etau.ensure_package("umap-learn")
)


logger = logging.getLogger(__name__)


# @todo optimize this
_DEFAULT_MODEL_NAME = "inception-v3-imagenet-torch"
_DEFAULT_BATCH_SIZE = 16


def compute_visualization(
    samples,
    embeddings,
    patches_field,
    embeddings_field,
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

    if model is None and embeddings_field is None and embeddings is None:
        model = foz.load_zoo_model(_DEFAULT_MODEL_NAME)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

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
        embeddings = [embeddings[_id] for _id in samples._get_sample_ids()]
        embeddings = [e for e in embeddings if e is not None and e.size > 0]
        embeddings = np.concatenate(embeddings, axis=0)

    logger.info("Generating visualization...")
    points = brain_method.fit(embeddings)

    results = VisualizationResults(samples, embeddings, points, config)
    brain_method.save_run_results(samples, brain_key, results)

    return results


class VisualizationResults(fob.BrainResults):
    """Class for visualizing the results of :meth:`compute_visualization`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` for
            which this visualization was computed
        embeddings: a ``num_samples x num_features`` array of embeddings
        points: a ``num_samples x num_dims`` array of visualization points
        config: the :class:`VisualizationConfig` used to generate the points
    """

    def __init__(self, samples, embeddings, points, config):
        self._samples = samples
        self.embeddings = embeddings
        self.points = points
        self.config = config

    def plot(
        self,
        field=None,
        labels=None,
        classes=None,
        session=None,
        marker_size=None,
        cmap=None,
        ax=None,
        ax_equal=True,
        figsize=None,
        style="seaborn-ticks",
        block=False,
        **kwargs,
    ):
        """Generates a scatterplot of the visualization results.

        This method supports 2D or 3D visualizations, but interactive point
        selection is only aviailable in 2D.

        Args:
            field (None): a sample field or ``embedded.field.name`` to use to
                color the points. Can be numeric or strings
            labels (None): a list of numeric or string values to use to color
                the points
            classes (None): an optional list of classes whose points to plot.
                Only applicable when ``labels`` contains strings
            session (None): a :class:`fiftyone.core.session.Session` object to
                link with the interactive plot. Only supported in 2D
            marker_size (None): the marker size to use
            cmap (None): a colormap recognized by ``matplotlib``
            ax (None): an optional matplotlib axis to plot in
            ax_equal (True): whether to set ``axis("equal")``
            figsize (None): an optional ``(width, height)`` for the figure, in
                inches
            style ("seaborn-ticks"): a style to use for the plot
            block (False): whether to block execution when the plot is
                displayed via ``matplotlib.pyplot.show(block=block)``
            **kwargs: optional keyword arguments for matplotlib's ``scatter()``

        Returns:
            a :class:`fiftyone.utils.plot.selector.PointSelector` if this is a
            2D visualization, else None
        """
        return foup.scatterplot(
            self.points,
            samples=self._samples,
            label_field=self.config.patches_field,
            field=field,
            labels=labels,
            classes=classes,
            session=session,
            marker_size=marker_size,
            cmap=cmap,
            ax=ax,
            ax_equal=ax_equal,
            figsize=figsize,
            style=style,
            block=block,
            **kwargs,
        )

    @classmethod
    def _from_dict(cls, d, samples):
        embeddings = np.array(d["embeddings"])
        points = np.array(d["points"])
        config = VisualizationConfig.from_dict(d["config"])
        return cls(samples, embeddings, points, config)


class VisualizationConfig(fob.BrainMethodConfig):
    """Base class for configuring :class:`Visualization` instances.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        patches_field (None): the sample field defining the patches we're
            visualizing
        num_dims (2): the dimension of the visualization space
    """

    def __init__(
        self, embeddings_field=None, patches_field=None, num_dims=2, **kwargs
    ):
        super().__init__(**kwargs)
        self.embeddings_field = embeddings_field
        self.patches_field = patches_field
        self.num_dims = num_dims


class Visualization(fob.BrainMethod):
    """Base class for embedding visualization methods.

    Args:
        config: a :class:`VisualizationConfig`
    """

    def fit(self, embeddings):
        """Computes visualization coordinates for the given embeddings.

        Args:
            embeddings: a ``num_samples x num_features`` array of embeddings

        Returns:
            a ``num_samples x num_dims`` array of coordinates
        """
        raise NotImplementedError("subclass must implement visualize()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass


class TSNEVisualizationConfig(VisualizationConfig):
    """Configuration for a :class:`TSNEVisualization`.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        patches_field (None): the sample field defining the patches we're
            visualizing
        num_dims (2): the dimension of the visualization space
        pca_dims (50): the number of PCA dimensions to compute prior to running
            t-SNE. It is highly recommended to reduce the number of dimensions
            to a reasonable number (e.g. 50) before running t-SNE, as this will
            suppress some noise and speed up the computation of pairwise
            distances between samples
        metric ("euclidean"): the metric to use when calculating distance
            between embeddings. Must be a supported value for the ``metric``
            argument of ``scipy.spatial.distance.pdist``
        perplexity (30.0): the perplexity to use. Perplexity is related to the
            number of nearest neighbors that is used in other manifold learning
            algorithms. Larger datasets usually require a larger perplexity.
            Typical values are in ``[5, 50]``
        learning_rate (200.0): the learning rate to use. Typical values are
            in ``[10, 1000]``. If the learning rate is too high, the data may
            look like a ball with any point approximately equidistant from its
            nearest neighbours. If the learning rate is too low, most points
            may look compressed in a dense cloud with few outliers. If the cost
            function gets stuck in a bad local minimum increasing the learning
            rate may help
        max_iters (1000): the maximum number of iterations to run. Should be at
            least 250
        seed (None): a random seed
        verbose (False): whether to log progress
    """

    def __init__(
        self,
        embeddings_field=None,
        patches_field=None,
        num_dims=2,
        pca_dims=50,
        metric="euclidean",
        perplexity=30.0,
        learning_rate=200.0,
        max_iters=1000,
        seed=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            patches_field=patches_field,
            num_dims=num_dims,
            **kwargs,
        )
        self.pca_dims = pca_dims
        self.metric = metric
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.seed = seed
        self.verbose = verbose

    @property
    def method(self):
        return "tsne"


class TSNEVisualization(Visualization):
    """t-distributed Stochastic Neighbor Embedding (t-SNE) embedding
    visualization.

    Args:
        config: a :class:`TSNEVisualizationConfig`
    """

    def fit(self, embeddings):
        if self.config.pca_dims is not None:
            _pca = skd.PCA(
                n_components=self.config.pca_dims,
                svd_solver="randomized",
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


class UMAPVisualizationConfig(VisualizationConfig):
    """Configuration for a :class:`UMAPVisualization`.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        patches_field (None): the sample field defining the patches we're
            visualizing
        num_dims (2): the dimension of the visualization space
        num_neighbors (15): the number of neighboring points used in local
            approximations of manifold structure. Larger values will result i
             more global structure being preserved at the loss of detailed
             local structure. Typical values are in ``[5, 50]``
        metric ("euclidean"): the metric to use when calculating distance
            between embeddings. See `https://github.com/lmcinnes/umap`_ for
            supported values
        min_dist (0.1): the effective minimum distance between embedded
            points. This controls how tightly the embedding is allowed compress
            points together. Larger values ensure embedded points are more
            evenly distributed, while smaller values allow the algorithm to
            optimise more accurately with regard to local structure. Typical
            values are in ``[0.001, 0.5]``
        seed (None): a random seed
        verbose (False): whether to log progress
    """

    def __init__(
        self,
        embeddings_field=None,
        patches_field=None,
        num_dims=2,
        num_neighbors=15,
        metric="euclidean",
        min_dist=0.1,
        seed=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            patches_field=patches_field,
            num_dims=num_dims,
            **kwargs,
        )
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.min_dist = min_dist
        self.seed = seed
        self.verbose = verbose

    @property
    def method(self):
        return "umap"


class UMAPVisualization(Visualization):
    """Uniform Manifold Approximation and Projection (UMAP) embedding
    visualization.

    Args:
        config: a :class:`UMAPVisualizationConfig`
    """

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


def _parse_config(
    config, embeddings_field, patches_field, method, num_dims, **kwargs
):
    if config is not None:
        return config

    if method is None or method == "tsne":
        config_cls = TSNEVisualizationConfig
    elif method == "umap":
        config_cls = UMAPVisualizationConfig
    else:
        raise ValueError("Unsupported method '%s'" % method)

    return config_cls(
        embeddings_field=embeddings_field,
        patches_field=patches_field,
        num_dims=num_dims,
        **kwargs,
    )
