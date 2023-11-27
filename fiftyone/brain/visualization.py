"""
Visualization interface.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.utils as fou

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")
fbv = fou.lazy_import("fiftyone.brain.internal.core.visualization")


class VisualizationResults(fob.BrainResults):
    """Class storing the results of
    :meth:`fiftyone.brain.compute_visualization`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`VisualizationConfig` used
        brain_key: the brain key
        points: a ``num_points x num_dims`` array of visualization points
        sample_ids (None): a ``num_points`` array of sample IDs
        label_ids (None): a ``num_points`` array of label IDs, if applicable
        backend (None): a :class:`Visualization` backend
    """

    def __init__(
        self,
        samples,
        config,
        brain_key,
        points,
        sample_ids=None,
        label_ids=None,
        backend=None,
    ):
        super().__init__(samples, config, brain_key, backend=backend)

        if sample_ids is None:
            sample_ids, label_ids = fbu.get_ids(
                samples,
                patches_field=config.patches_field,
                data=points,
                data_type="points",
            )

        self.points = points
        self.sample_ids = sample_ids
        self.label_ids = label_ids

        self._last_view = None
        self._curr_view = None
        self._curr_points = None
        self._curr_sample_ids = None
        self._curr_label_ids = None
        self._curr_keep_inds = None
        self._curr_good_inds = None

        self.use_view(samples)

    def __enter__(self):
        self._last_view = self.view
        return self

    def __exit__(self, *args):
        self.use_view(self._last_view)
        self._last_view = None

    @property
    def config(self):
        """The :class:`VisualizationConfig` for the results."""
        return self._config

    @property
    def index_size(self):
        """The number of active points in the index.

        If :meth:`use_view` has been called to restrict the index, this
        property will reflect the size of the active index.
        """
        return len(self._curr_sample_ids)

    @property
    def total_index_size(self):
        """The total number of data points in the index.

        If :meth:`use_view` has been called to restrict the index, this value
        may be larger than the current :meth:`index_size`.
        """
        return len(self.points)

    @property
    def missing_size(self):
        """The total number of data points in :meth:`view` that are missing
        from this index.

        This property is only applicable when :meth:`use_view` has been called,
        and it will be ``None`` if no data points are missing.
        """
        good = self._curr_good_inds

        if good is None:
            return None

        return good.size - np.count_nonzero(good)

    @property
    def current_points(self):
        """The currently active points in the index.

        If :meth:`use_view` has been called, this may be a subset of the full
        index.
        """
        return self._curr_points

    @property
    def current_sample_ids(self):
        """The sample IDs of the currently active points in the index.

        If :meth:`use_view` has been called, this may be a subset of the full
        index.
        """
        return self._curr_sample_ids

    @property
    def current_label_ids(self):
        """The label IDs of the currently active points in the index, or
        ``None`` if not applicable.

        If :meth:`use_view` has been called, this may be a subset of the full
        index.
        """
        return self._curr_label_ids

    @property
    def view(self):
        """The :class:`fiftyone.core.collections.SampleCollection` against
        which results are currently being generated.

        If :meth:`use_view` has been called, this view may be different than
        the collection on which the full index was generated.
        """
        return self._curr_view

    def use_view(
        self, sample_collection, allow_missing=True, warn_missing=False
    ):
        """Restricts the index to the provided view.

        Subsequent calls to methods on this instance will only contain results
        from the specified view rather than the full index.

        Use :meth:`clear_view` to reset to the full index. Or, equivalently,
        use the context manager interface as demonstrated below to
        automatically reset the view when the context exits.

        Example usage::

            import fiftyone as fo
            import fiftyone.brain as fob
            import fiftyone.zoo as foz

            dataset = foz.load_zoo_dataset("quickstart")

            results = fob.compute_visualization(dataset)
            print(results.index_size)  # 200

            view = dataset.take(50)

            with results.use_view(view):
                print(results.index_size)  # 50

                plot = results.visualize()
                plot.show()

        Args:
            sample_collection: a
                :class:`fiftyone.core.collections.SampleCollection`
            allow_missing (True): whether to allow the provided collection to
                contain data points that this index does not contain (True) or
                whether to raise an error in this case (False)
            warn_missing (False): whether to log a warning if the provided
                collection contains data points that this index does not
                contain

        Returns:
            self
        """
        sample_ids, label_ids, keep_inds, good_inds = fbu.filter_ids(
            sample_collection,
            self.sample_ids,
            self.label_ids,
            patches_field=self._config.patches_field,
            allow_missing=allow_missing,
            warn_missing=warn_missing,
        )

        if keep_inds is not None:
            points = self.points[keep_inds, :]
        else:
            points = self.points

        self._curr_view = sample_collection
        self._curr_points = points
        self._curr_sample_ids = sample_ids
        self._curr_label_ids = label_ids
        self._curr_keep_inds = keep_inds
        self._curr_good_inds = good_inds

        return self

    def clear_view(self):
        """Clears the view set by :meth:`use_view`, if any.

        Subsequent operations will be performed on the full index.
        """
        self.use_view(self._samples)

    def values(self, path_or_expr):
        """Extracts a flat list of values from the given field or expression
        corresponding to the current :meth:`view`.

        This method always returns values in the same order as
        :meth:`current_points`, :meth:`current_sample_ids`, and
        :meth:`current_label_ids`.

        Args:
            path_or_expr: the values to extract, which can be:

                -   the name of a sample field or ``embedded.field.name`` from
                    which to extract numeric or string values
                -   a :class:`fiftyone.core.expressions.ViewExpression`
                    defining numeric or string values to compute via
                    :meth:`fiftyone.core.collections.SampleCollection.values`

        Returns:
            a list of values
        """
        return fbv.values(self, path_or_expr)

    def visualize(
        self,
        labels=None,
        sizes=None,
        classes=None,
        backend="plotly",
        **kwargs,
    ):
        """Generates an interactive scatterplot of the visualization results
        for the current :meth:`view`.

        This method supports 2D or 3D visualizations, but interactive point
        selection is only available in 2D.

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
                -   a :class:`fiftyone.core.expressions.ViewExpression`
                    defining numeric or string values to compute via
                    :meth:`fiftyone.core.collections.SampleCollection.values`
                -   a list or array-like of numeric or string values
                -   a list of lists of numeric or string values, if the data in
                    this visualization corresponds to a label list field like
                    :class:`fiftyone.core.labels.Detections`

            sizes (None): data to use to scale the sizes of the points. Can be
                any of the following:

                -   the name of a sample field or ``embedded.field.name`` from
                    which to extract numeric values
                -   a :class:`fiftyone.core.expressions.ViewExpression`
                    defining numeric values to compute via
                    :meth:`fiftyone.core.collections.SampleCollection.values`
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
        return fbv.visualize(
            self,
            labels=labels,
            sizes=sizes,
            classes=classes,
            backend=backend,
            **kwargs,
        )

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        points = np.array(d["points"])

        sample_ids = d.get("sample_ids", None)
        if sample_ids is not None:
            sample_ids = np.array(sample_ids)

        label_ids = d.get("label_ids", None)
        if label_ids is not None:
            label_ids = np.array(label_ids)

        return cls(
            samples,
            config,
            brain_key,
            points,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )


class VisualizationConfig(fob.BrainMethodConfig):
    """Base class for configuring visualization methods.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        num_dims=2,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.num_dims = num_dims
        super().__init__(**kwargs)

    @property
    def type(self):
        return "visualization"

    @property
    def run_cls(self):
        run_cls_name = self.__class__.__name__[: -len("Config")]
        return getattr(fbv, run_cls_name)


class UMAPVisualizationConfig(VisualizationConfig):
    """Configuration for Uniform Manifold Approximation and Projection (UMAP)
    embedding visualization.

    See https://github.com/lmcinnes/umap for more information about the
    supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
        num_neighbors (15): the number of neighboring points used in local
            approximations of manifold structure. Larger values will result in
            more global structure being preserved at the loss of detailed local
            structure. Typical values are in ``[5, 50]``
        metric ("euclidean"): the metric to use when calculating distance
            between embeddings. See the UMAP documentation for supported values
        min_dist (0.1): the effective minimum distance between embedded
            points. This controls how tightly the embedding is allowed compress
            points together. Larger values ensure embedded points are more
            evenly distributed, while smaller values allow the algorithm to
            optimise more accurately with regard to local structure. Typical
            values are in ``[0.001, 0.5]``
        seed (None): a random seed
        verbose (True): whether to log progress
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        num_dims=2,
        num_neighbors=15,
        metric="euclidean",
        min_dist=0.1,
        seed=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
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
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
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
        verbose (True): whether to log progress
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        num_dims=2,
        pca_dims=50,
        svd_solver="randomized",
        metric="euclidean",
        perplexity=30.0,
        learning_rate=200.0,
        max_iters=1000,
        seed=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
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
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
        svd_solver ("randomized"): the SVD solver to use. Consult the sklearn
            docmentation for details
        seed (None): a random seed
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        num_dims=2,
        svd_solver="randomized",
        seed=None,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            num_dims=num_dims,
            **kwargs,
        )
        self.svd_solver = svd_solver
        self.seed = seed

    @property
    def method(self):
        return "pca"


class ManualVisualizationConfig(VisualizationConfig):
    """Configuration for manually-provided low-dimensional visualizations.

    Args:
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
    """

    def __init__(self, patches_field=None, num_dims=2, **kwargs):
        super().__init__(
            patches_field=patches_field, num_dims=num_dims, **kwargs
        )

    @property
    def method(self):
        return "manual"
