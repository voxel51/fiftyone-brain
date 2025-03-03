"""
Visualization interface.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import deepcopy
import inspect
import logging

import numpy as np
import sklearn.decomposition as skd
import sklearn.manifold as skm

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.brain as fob
import fiftyone.core.expressions as foe
import fiftyone.core.fields as fof
import fiftyone.core.plots as fop
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")

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
    similarity_index,
    model,
    model_kwargs,
    force_square,
    alpha,
    batch_size,
    num_workers,
    skip_failures,
    progress,
    point_field=None,
    index_points=False,
    **kwargs,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)
    point_field = _get_point_field(
        brain_key, points, point_field, index_points
    )
    points = None if point_field else points
    points_not_provided = points is None
    point_field_index = f"{point_field}_index" if point_field else None

    if method == "manual" and points_not_provided:
        raise ValueError(
            "You must provide your own `points` when `method='manual'`"
        )

    if points is not None and not point_field:
        points_not_provided = False
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

    if etau.is_str(point_field):
        # parse_embeddings_field can be used for point_field as well
        point_field, points_exist = fbu.parse_embeddings_field(
            samples, point_field, patches_field
        )

    if etau.is_str(similarity_index):
        similarity_index = samples.load_brain_results(similarity_index)

    if (
        model is None
        and points_not_provided
        and embeddings is None
        and similarity_index is None
        and not embeddings_exist
    ):
        model = _DEFAULT_MODEL
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = _parse_config(
        method,
        embeddings_field=embeddings_field,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
        patches_field=patches_field,
        num_dims=num_dims,
        point_field=point_field,
        point_field_index=point_field_index,
        **kwargs,
    )

    brain_method = config.build()
    brain_method.ensure_requirements()

    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    if points_not_provided:
        embeddings, sample_ids, label_ids = fbu.get_embeddings(
            samples,
            model=model,
            model_kwargs=model_kwargs,
            patches_field=patches_field,
            embeddings_field=embeddings_field,
            embeddings=embeddings,
            similarity_index=similarity_index,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=progress,
        )

        logger.info("Generating visualization...")
        points = brain_method.fit(embeddings)
        if point_field:
            logger.info(f"Indexing visualization on field '{point_field}'")
            min_val, max_val = _get_min_max(points)
            _define_point_field(samples, point_field, min_val, max_val)
            _populate_point_field(samples, point_field, points, progress)
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
    # TODO: handle point_field - could just use view.values()
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


def _parse_config(name, **kwargs):
    if name is None:
        name = fb.brain_config.default_visualization_method

    if inspect.isclass(name):
        return name(**kwargs)

    methods = fb.brain_config.visualization_methods

    if name not in methods:
        raise ValueError(
            "Unsupported method '%s'. The available methods are %s"
            % (name, sorted(methods.keys()))
        )

    params = deepcopy(methods[name])

    config_cls = kwargs.pop("config_cls", None)

    if config_cls is None:
        config_cls = params.pop("config_cls", None)

    if config_cls is None:
        raise ValueError(
            "Visualization method '%s' has no `config_cls`" % name
        )

    if etau.is_str(config_cls):
        config_cls = etau.get_class(config_cls)

    params.update(**kwargs)
    return config_cls(**params)


def _get_dimension(points):
    if isinstance(points, dict):
        points = next(iter(points.values()), None)

    if isinstance(points, list):
        points = next(iter(points), None)

    if points is None:
        return 2

    return points.shape[-1]


def _get_point_field(
    brain_key, points_arg, point_field_arg, use_default_point_field=False
):
    if points_arg and point_field_arg:
        raise ValueError("You cannot provide both `points` and `point_field`")

    if etau.is_str(points_arg):
        return points_arg

    if etau.is_str(point_field_arg):
        return point_field_arg

    if use_default_point_field:
        return f"_{brain_key}_point"

    return None


def _get_min_max(points):
    flat_points = points.flatten()
    min, max = float(np.min(flat_points)), float(np.max(flat_points))
    return min, max


def _define_point_field(samples, point_field, min_val, max_val):
    dataset = samples._dataset
    dataset.add_sample_field(
        point_field, fof.ListField, subfield=fof.FloatField
    )
    try:
        dataset.delete_index(point_field)
    except:
        # TODO: should we log this?
        #       should we catch other errors and re-raise?
        pass
    dataset.create_index(
        [(point_field, "2d")], min=min_val, max=max_val, name=point_field
    )


def _populate_point_field(samples, point_field, points, progress=True):
    samples_and_points = zip(samples, points)
    iterator = (
        fou.ProgressBar(samples_and_points) if progress else samples_and_points
    )
    for sample, point in iterator:
        sample.set_field(point_field, point.tolist())
        sample.save()


class VisualizationResults(fob.BrainResults):
    """Class storing the results of
    :meth:`fiftyone.brain.compute_visualization`.

    NOTE: either `points` or `point_field` must be provided.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`VisualizationConfig` used
        brain_key: the brain key
        points (None): an optional ``num_points x num_dims`` array of visualization points
        point_field (None): the name of an optional field where points are stored and indexed
        sample_ids (None): a ``num_points`` array of sample IDs
        label_ids (None): a ``num_points`` array of label IDs, if applicable
        backend (None): a :class:`Visualization` backend
    """

    def __init__(
        self,
        samples,
        config,
        brain_key,
        points=None,
        point_field=None,
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

        if points is None and point_field is None:
            raise ValueError(
                "Either `points` or `point_field` must be provided"
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
        if self.point_field:
            return len(self.view)
        return len(self._curr_sample_ids)

    @property
    def total_index_size(self):
        """The total number of data points in the index.

        If :meth:`use_view` has been called to restrict the index, this value
        may be larger than the current :meth:`index_size`.
        """
        print(self.point_field)
        if self.point_field:
            return len(self._curr_view)
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
        return values(self, path_or_expr)

    def get_points(self):
        """Returns the current points in the index.

        If :meth:`use_view` has been called, this method will return the
        currently active points in the index. Otherwise, it will return the
        full index.

        Returns:
            a ``num_points x num_dims`` array of visualization points
        """
        if self.point_field:
            return self.view.values(self.point_field)

        return self.points

    @property
    def point_field(self):
        """The name of the field in which the visualization points are stored,
        or ``None`` if the points are stored directly in the results.
        """
        return self._config.point_field

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
        return visualize(
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
        similarity_index (None): the similarity index containing the
            embeddings, if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
        point_field (None): the name of a field in which to store the
            visualization points, if desired
        point_field_index (None): the name of the index field for the
    """

    def __init__(
        self,
        embeddings_field=None,
        similarity_index=None,
        model=None,
        model_kwargs=None,
        patches_field=None,
        num_dims=2,
        point_field=None,
        point_field_index=None,
        **kwargs,
    ):
        if similarity_index is not None and not etau.is_str(similarity_index):
            similarity_index = similarity_index.key

        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.similarity_index = similarity_index
        self.model = model
        self.model_kwargs = model_kwargs
        self.patches_field = patches_field
        self.num_dims = num_dims
        self.point_field = point_field
        self.point_field_index = point_field_index
        super().__init__(**kwargs)

    @property
    def type(self):
        return "visualization"


class Visualization(fob.BrainMethod):
    def fit(self, embeddings):
        raise NotImplementedError("subclass must implement fit()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        if self.config.point_field is not None:
            # TODO: this doesn't seem to delete the point field!?
            fields.append(self.config.point_field)

        return fields

    def cleanup(self, samples, key):
        dataset = samples._dataset
        point_field_index = self.config.point_field_index
        has_index = point_field_index in dataset.list_indexes()
        if has_index:
            dataset.delete_index(self.config.point_field)


class UMAPVisualizationConfig(VisualizationConfig):
    """Configuration for Uniform Manifold Approximation and Projection (UMAP)
    embedding visualization.

    See https://github.com/lmcinnes/umap for more information about the
    supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        similarity_index (None): the similarity index containing the
            embeddings, if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
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
        similarity_index=None,
        model=None,
        model_kwargs=None,
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
            similarity_index=similarity_index,
            model=model,
            model_kwargs=model_kwargs,
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


class TSNEVisualizationConfig(VisualizationConfig):
    """Configuration for t-distributed Stochastic Neighbor Embedding (t-SNE)
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    for more information about the supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        similarity_index (None): the similarity index containing the
            embeddings, if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
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
        similarity_index=None,
        model=None,
        model_kwargs=None,
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
            similarity_index=similarity_index,
            model=model,
            model_kwargs=model_kwargs,
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


class PCAVisualizationConfig(VisualizationConfig):
    """Configuration for principal component analysis (PCA) embedding
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    for more information about the supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        similarity_index (None): the similarity index containing the
            embeddings, if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
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
        similarity_index=None,
        model=None,
        model_kwargs=None,
        patches_field=None,
        num_dims=2,
        svd_solver="randomized",
        seed=None,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            similarity_index=similarity_index,
            model=model,
            model_kwargs=model_kwargs,
            patches_field=patches_field,
            num_dims=num_dims,
            **kwargs,
        )
        self.svd_solver = svd_solver
        self.seed = seed

    @property
    def method(self):
        return "pca"


class PCAVisualization(Visualization):
    def fit(self, embeddings):
        _pca = skd.PCA(
            n_components=self.config.num_dims,
            svd_solver=self.config.svd_solver,
            random_state=self.config.seed,
        )
        return _pca.fit_transform(embeddings)


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


class ManualVisualization(Visualization):
    def fit(self, embeddings):
        raise NotImplementedError(
            "The low-dimensional representation must be manually provided "
            "when using this method"
        )
