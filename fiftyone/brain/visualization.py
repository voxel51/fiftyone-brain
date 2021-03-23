"""
Public visualization interface.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.plots as fop


_INTERNAL_MODULE = "fiftyone.brain.internal.core.visualization"


class VisualizationResults(fob.BrainResults):
    """Class for visualizing the results of :meth:`compute_visualization`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` for
            which this visualization was computed
        points: a ``num_samples x num_dims`` array of visualization points
        config: the :class:`VisualizationConfig` used to generate the points
    """

    def __init__(self, samples, points, config):
        self._samples = samples
        self.points = points
        self.config = config

    def visualize(
        self,
        labels=None,
        sizes=None,
        classes=None,
        backend="plotly",
        **kwargs,
    ):
        """Generates an interactive scatterplot of the visualization results.

        This method supports 2D or 3D visualizations, but interactive point
        selection is only aviailable in 2D.

        You can use the ``labels`` parameters to define a coloring for the
        points, and you can use the ``sizes`` parameter to scale the sizes of
        the points.

        You can attach plots generated by this method to an App session via its
        :attr:`fiftyone.core.session.Session.plots` attribute, which will
        automatically sync the session's view with the currently selected
        points in the plot.

        Args:
            labels (None): data to use to color the points. Can be any of the
                following:

                -   the name of a sample field or ``embedded.field.name`` from
                    which to extract numeric or string values
                -   a list or array-like of numeric or string values
                -   a list of lists of numeric or string values, if the data in
                    this visualization corresponds to a label list field like
                    :class:`fiftyone.core.labels.Detections`

            sizes (None): data to use to scale the sizes of the points. Can be
                any of the following:

                -   the name of a sample field or ``embedded.field.name`` from
                    which to extract numeric values
                -   a list or array-like of numeric values
                -   a list of lists of numeric values, if the data in this
                    visualization corresponds to a label list field like
                    :class:`fiftyone.core.labels.Detections`

            classes (None): an optional list of classes whose points to plot.
                Only applicable when ``labels`` contains strings
            backend ("plotly"): the plotting backend to use. Supported values
                are ``("plotly", "matplotlib")``
            **kwargs: keyword arguments for the backend plotting method:

                -   "plotly" backend: :meth:`fiftyone.core.plots.plotly.scatterplot`
                -   "matplotlib" backend: :meth:`fiftyone.core.plots.matplotlib.scatterplot`

        Returns:
            an :class:`fiftyone.core.plots.base.InteractivePlot`
        """
        return fop.scatterplot(
            self.points,
            samples=self._samples,
            label_field=self.config.patches_field,
            labels=labels,
            sizes=sizes,
            classes=classes,
            backend=backend,
            **kwargs,
        )

    @classmethod
    def _from_dict(cls, d, samples):
        points = np.array(d["points"])
        config = VisualizationConfig.from_dict(d["config"])
        return cls(samples, points, config)


class VisualizationConfig(fob.BrainMethodConfig):
    """Base class for configuring visualization methods.

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

    @property
    def run_cls(self):
        visualization_cls_name = self.__class__.__name__[: -len("Config")]
        return etau.get_class(_INTERNAL_MODULE + "." + visualization_cls_name)


class UMAPVisualizationConfig(VisualizationConfig):
    """Configuration for Uniform Manifold Approximation and Projection (UMAP)
    embedding visualization.

    See https://github.com/lmcinnes/umap for more information about the
    supported parameters.

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
            between embeddings. See the UMAP documentation for supported values
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


class TSNEVisualizationConfig(VisualizationConfig):
    """Configuration for t-distributed Stochastic Neighbor Embedding (t-SNE)
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    for more information about the supported parameters.

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
        svd_solver ("randomized"): the SVD solver to use when performing PCA.
            Consult the sklearn docmentation for details
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
        svd_solver="randomized",
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
        self.svd_solver = svd_solver
        self.metric = metric
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.seed = seed
        self.verbose = verbose

    @property
    def method(self):
        return "tsne"


class PCAVisualizationConfig(VisualizationConfig):
    """Configuration for principal component analysis (PCA) embedding
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    for more information about the supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        patches_field (None): the sample field defining the patches we're
            visualizing
        num_dims (2): the dimension of the visualization space
        svd_solver ("randomized"): the SVD solver to use. Consult the sklearn
            docmentation for details
        seed (None): a random seed
    """

    def __init__(
        self,
        embeddings_field=None,
        patches_field=None,
        num_dims=2,
        svd_solver="randomized",
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            patches_field=patches_field,
            num_dims=num_dims,
            **kwargs,
        )
        self.svd_solver = svd_solver

    @property
    def method(self):
        return "pca"
