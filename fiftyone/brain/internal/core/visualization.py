"""
Visualization methods.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import itertools
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import sklearn.decomposition as skd
import sklearn.manifold as skm

import eta.core.serial as etas
import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
from fiftyone.core.view import DatasetView
import fiftyone.zoo as foz

from .selector import PointSelector

umap = fou.lazy_import(
    "umap", callback=lambda: etau.ensure_package("umap-learn")
)


logger = logging.getLogger(__name__)


_DEFAULT_MODEL_NAME = "inception-v3-imagenet-torch"


def compute_visualization(
    samples,
    embeddings=None,
    patches_field=None,
    embeddings_field=None,
    brain_key=None,
    num_dims=2,
    method="tsne",
    config=None,
    model=None,
    batch_size=None,
    force_square=False,
    alpha=None,
    **kwargs,
):
    """Computes a low-dimensional representation of the samples' media or their
    patches that can be interactively visualized and manipulated via the
    returned :class:`VisualizationResults` object.

    If no ``embeddings``, ``embeddings_field``, or ``model`` is provided, a
    default model is used to generate embeddings.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        embeddings (None): a ``num_samples x num_dims`` array of embeddings,
            or, if a ``patches_field`` is specified,  a dict mapping sample IDs
            to ``num_patches x num_dims`` arrays of patch embeddings
        patches_field (None): a sample field defining the image patches in each
            sample that have been/will be embedded
        embeddings_field (None): the name of a field containing embeddings to
            use
        brain_key (None): a brain key under which to store the results of this
            visualization
        num_dims (2): the dimension of the visualization space
        method ("tsne"): the dimensionality-reduction method to use. Supported
            values are ``("tsne", "umap")``
        config (None): a :class:`VisualizationConfig` specifying the parameters
             to use. If provided, takes precedence over other parameters
        model (None): a :class:`fiftyone.core.models.Model` to use to generate
            embeddings
        batch_size (None): an optional batch size to use when computing
            embeddings. Only applicable when a ``model`` is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``patches_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, \infty)``. If provided, the
            length and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 1.1`` to expand the boxes by 10%, and set ``alpha = 0.9``
            to contract the boxes by 10%. Only applicable when a ``model`` and
            ``patches_field`` are specified
        **kwargs: optional keyword arguments for the constructor of the
            :class:`VisualizationConfig` being used

    Returns:
        a :class:`VisualizationResults`
    """
    fov.validate_collection(samples)

    if model is None and embeddings_field is None and embeddings is None:
        model = foz.load_zoo_model(_DEFAULT_MODEL_NAME)

    config = _parse_config(
        config, embeddings_field, patches_field, method, num_dims, **kwargs
    )
    brain_method = config.build()
    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    if model is not None:
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
            figsize (None): an optional ``(width, height)`` for the figure, in
                inches
            style ("seaborn-ticks"): a style to use for the plot
            block (False): whether to block execution when the plot is
                displayed via ``matplotlib.pyplot.show(block=block)``
            **kwargs: optional keyword arguments for matplotlib's ``scatter()``

        Returns:
            a :class:`PointSelector` if this is a 2D visualization, else None
        """
        if self.config.num_dims not in {2, 3}:
            raise ValueError(
                "This method only supports 2D or 3D visualization"
            )

        if session is not None and self.config.num_dims != 2:
            logger.warning("Interactive selection is only supported in 2D")

        if field is not None:
            labels = self._samples.values(field)

        if labels and isinstance(labels[0], (list, tuple)):
            labels = list(itertools.chain.from_iterable(labels))

        if labels is not None:
            if len(labels) != len(self.points):
                raise ValueError(
                    "Number of labels (%d) does not match number of points "
                    "(%d). You may have missing embeddings and/or labels that "
                    "you need to omit from your view before visualizing"
                    % (len(labels), len(self.points))
                )

        with plt.style.context(style):
            ax, coll, inds = _plot_scatter(
                self.points,
                labels=labels,
                classes=classes,
                marker_size=marker_size,
                cmap=cmap,
                ax=ax,
                figsize=figsize,
                **kwargs,
            )

        if self.config.num_dims != 2:
            plt.tight_layout()
            plt.show(block=block)
            return None

        sample_ids = None
        object_ids = None
        if self.config.patches_field is not None:
            object_ids = np.array(
                _get_object_ids(self._samples, self.config.patches_field)
            )
            if inds is not None:
                object_ids = object_ids[inds]
        else:
            sample_ids = np.array(self._samples._get_sample_ids())
            if inds is not None:
                sample_ids = sample_ids[inds]

        if session is not None:
            if isinstance(self._samples, DatasetView):
                session.view = self._samples
            else:
                session.dataset = self._samples

        selector = PointSelector(
            ax,
            coll,
            session=session,
            sample_ids=sample_ids,
            object_ids=object_ids,
            object_field=self.config.patches_field,
        )

        plt.tight_layout()
        plt.show(block=block)

        return selector

    # pylint: disable=no-member
    @classmethod
    def load_run_results(cls, samples, key):
        results = super().load_run_results(samples, key)
        view = cls.load_run_view(samples, key)
        results._samples = view
        return results

    # pylint: disable=no-member
    @classmethod
    def _from_dict(cls, d):
        embeddings = etas.deserialize_numpy_array(d["embeddings"])
        points = etas.deserialize_numpy_array(d["points"])
        config = VisualizationConfig.from_dict(d["config"])
        return cls(None, embeddings, points, config)


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


def _plot_scatter(
    points,
    labels=None,
    classes=None,
    marker_size=None,
    cmap=None,
    ax=None,
    figsize=None,
    **kwargs,
):
    if labels is not None:
        points, values, classes, inds, categorical = _parse_data(
            points, labels, classes
        )
    else:
        values, classes, inds, categorical = None, None, None, None

    scatter_3d = points.shape[1] == 3

    if ax is None:
        projection = "3d" if scatter_3d else None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)
    else:
        fig = ax.figure

    if cmap is None:
        cmap = "Spectral" if categorical else "viridis"

    cmap = plt.get_cmap(cmap)

    if categorical:
        boundaries = np.arange(0, len(classes) + 1)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    else:
        norm = None

    if marker_size is None:
        marker_size = 10 ** (4 - np.log10(points.shape[0]))
        marker_size = max(0.1, min(marker_size, 25))
        marker_size = round(marker_size, 0 if marker_size >= 1 else 1)

    args = [points[:, 0], points[:, 1]]
    if scatter_3d:
        args.append(points[:, 2])

    coll = ax.scatter(
        *args, c=values, s=marker_size, cmap=cmap, norm=norm, **kwargs,
    )

    if values is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right", size="5%", pad=0.1, axes_class=mpl.axes.Axes
        )

        if categorical:
            ticks = 0.5 + np.arange(0, len(classes))
            cbar = mpl.colorbar.ColorbarBase(
                cax,
                cmap=cmap,
                norm=norm,
                spacing="proportional",
                boundaries=boundaries,
                ticks=ticks,
            )
            cbar.set_ticklabels(classes)
        else:
            mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(values)
            fig.colorbar(mappable, cax=cax)

    ax.axis("equal")

    if figsize is not None:
        fig.set_size_inches(*figsize)

    return ax, coll, inds


def _parse_data(points, labels, classes):
    if not labels:
        return points, None, None, None, False

    if not etau.is_str(labels[0]):
        return points, labels, None, None, False

    if classes is None:
        classes = sorted(set(labels))

    values_map = {c: i for i, c in enumerate(classes)}
    values = np.array([values_map.get(l, -1) for l in labels])

    found = values >= 0
    if not np.all(found):
        points = points[found, :]
        values = values[found]
    else:
        found = None

    return points, values, classes, found, True


def _get_object_ids(samples, patches_field):
    label_type, id_path = samples._get_label_field_path(patches_field, "_id")
    if issubclass(label_type, (fol.Detection, fol.Polyline)):
        return [str(_id) for _id in samples.values(id_path)]

    if issubclass(label_type, (fol.Detections, fol.Polylines)):
        object_ids = samples.values(id_path)
        return [str(_id) for _id in itertools.chain.from_iterable(object_ids)]

    raise ValueError(
        "Patches field %s has unsupported type %s"
        % (patches_field, label_type)
    )
