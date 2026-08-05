"""
Visualization interface.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import base64
from copy import deepcopy
import inspect
import logging
import pickle
import zlib
from packaging import version

import numpy as np
import sklearn
import sklearn.decomposition as skd
import sklearn.manifold as skm

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.brain as fob
import fiftyone.core.dataset as fod
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

_REDUCER_APPLY_ERROR = (
    "Failed to apply this visualization's stored reducer. This typically "
    "means the run was computed in a different environment (e.g. a "
    "different Python, umap-learn, or numba version), in which case the "
    "stored reducer cannot be used here. Please recompute the "
    "visualization (e.g. via compute_visualization()) to enable "
    "incremental updates in this environment"
)


def compute_visualization(
    samples,
    patches_field,
    embeddings,
    points,
    create_index,
    points_field,
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

    if create_index and points_field is None:
        points_field = brain_key

    if points_field is not None and num_dims != 2:
        raise ValueError("`points_field` is only supported when `num_dims=2`")

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_data_field(
            samples,
            embeddings,
            patches_field=patches_field,
            data_type="embeddings",
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if points_field is not None:
        points_field, _ = fbu.parse_data_field(
            samples,
            points_field,
            patches_field=patches_field,
            data_type="points",
        )

    if etau.is_str(similarity_index):
        similarity_index = samples.load_brain_results(similarity_index)

    if (
        model is None
        and points is None
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
        points_field=points_field,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
        patches_field=patches_field,
        num_dims=num_dims,
        **kwargs,
    )

    brain_method = config.build()
    brain_method.ensure_requirements()

    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    reducer = None
    if points is None:
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
        points, reducer = brain_method.fit_reducer(embeddings)

        if config.method == "umap" and embeddings_field is None:
            logger.info(
                "Computed UMAP visualization without an embeddings field; "
                "incremental updates via add_samples() won't be available. "
                "Pass `embeddings=<field name>` next time to enable them"
            )
            reducer = None
    else:
        points, sample_ids, label_ids = fbu.parse_data(
            samples,
            patches_field=patches_field,
            data=points,
            data_type="points",
        )

    if points_field is not None:
        _generate_spatial_index(
            samples,
            points,
            points_field,
            sample_ids,
            label_ids=label_ids,
            patches_field=patches_field,
            create_index=create_index,
            progress=progress,
        )

    results = VisualizationResults(
        samples,
        config,
        brain_key,
        points,
        sample_ids=sample_ids,
        label_ids=label_ids,
        reducer=reducer,
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


def _generate_spatial_index(
    samples,
    points,
    points_field,
    sample_ids,
    label_ids=None,
    patches_field=None,
    create_index=True,
    progress=False,
):
    # Indexes are not currently usable on patch visualizations
    if create_index and patches_field is not None:
        create_index = False

    dataset = samples._root_dataset
    if patches_field is not None:
        _, points_field = dataset._get_label_field_path(
            patches_field, points_field
        )

    logger.info("Generating spatial index in field '%s'...", points_field)

    dataset.add_sample_field(
        points_field, fof.ListField, subfield=fof.FloatField
    )

    points = points.astype(float)

    if create_index:
        min_val, max_val = points.min(), points.max()
        dataset.create_index([(points_field, "2d")], min=min_val, max=max_val)

    points = points.tolist()
    if patches_field is not None:
        values = dict(zip(label_ids, points))
        dataset.set_label_values(points_field, values, progress=progress)
    else:
        values = dict(zip(sample_ids, points))
        dataset.set_values(
            points_field, values, key_field="id", progress=progress
        )


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
        reducer (None): the fitted dimensionality reduction model used to
            generate ``points``, retained to support incremental updates via
            :meth:`add_samples`. Not applicable to all methods
        reducer_blob (None): a base64-encoded pickled form of ``reducer``,
            used when reconstructing the results from a persisted run
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
        reducer=None,
        reducer_blob=None,
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

        self._reducer = reducer
        self._reducer_blob = reducer_blob

        self._last_view = None
        self._curr_view = None
        self._curr_points = None
        self._curr_sample_ids = None
        self._curr_label_ids = None
        self._curr_keep_inds = None
        self._curr_good_inds = None

        self.use_view(samples)

    @property
    def reducer_blob(self):
        if self._reducer_blob is None and self._reducer is not None:
            self._reducer_blob = _pickle_reducer(self._reducer)

        return self._reducer_blob

    def __enter__(self):
        self._last_view = self.view
        return self

    def __exit__(self, *args):
        self.use_view(self._last_view)
        self._last_view = None

    # Serialized compactly by serialize() rather than expanded into JSON
    # lists by the base class
    _ARRAY_FIELDS = ("points", "sample_ids", "label_ids")

    def attributes(self):
        attrs = [
            a for a in super().attributes() if a not in self._ARRAY_FIELDS
        ]
        attrs.append("reducer_blob")
        return attrs

    def serialize(self, reflective=False):
        """Serializes the results into a dictionary.

        The array-valued fields (points and ids) are stored as
        zlib-compressed, base64-encoded ``.npy`` bytes rather than JSON
        lists: for large runs the list encoding inflates the stored blob
        several-fold, and deserializing it materializes millions of
        transient Python objects. :meth:`_from_dict` accepts both
        encodings, so results written by earlier versions load unchanged.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the object
        """
        d = super().serialize(reflective=reflective)
        for name in self._ARRAY_FIELDS:
            arr = getattr(self, name)
            d[name] = (
                fou.serialize_numpy_array(np.asarray(arr), ascii=True)
                if arr is not None
                else None
            )

        return d

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

    @property
    def has_spatial_index(self):
        """Whether these results have a spatial index.

        Use :meth:`index_points` to add a spatial index to an existing set of
        visualization results.
        """
        return self.config.points_field is not None

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
        if isinstance(sample_collection, fod.Dataset):
            # A root dataset contains every index point by construction,
            # so skip the id aggregation that filter_ids would run just
            # to rediscover that (measured ~1.3s at 500K points; this
            # runs on every results load via __init__). Like the index
            # itself, this treats the run as a snapshot: samples deleted
            # since compute are not pruned here (view-scoped calls still
            # prune) — refreshing the run is the supported way to sync a
            # visualization with dataset changes
            self._curr_view = sample_collection
            self._curr_points = self.points
            self._curr_sample_ids = self.sample_ids
            self._curr_label_ids = self.label_ids
            self._curr_keep_inds = None
            self._curr_good_inds = None

            return self

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

    def index_points(
        self,
        points_field=None,
        create_index=True,
        progress=None,
    ):
        """Adds a spatial index for these visualization results to its
        dataset's samples.

        This method is useful if you want to add a spatial index to existing
        visualization results that don't yet have one.

        Spatial indexes are highly recommended for large datasets as they
        enable efficient querying when lassoing points in embeddings plots.

        Args:
            points_field (None): an optional field name in which to store the
                spatial index. The default is the result's ``brain_key``
            create_index (True): whether to create a database index for the
                points
            progress (None): whether to render a progress bar (True/False),
                use the default value ``fiftyone.config.show_progress_bars``
                (None), or a progress callback function to invoke instead
        """
        if points_field is None:
            if self.key is None:
                raise ValueError(
                    "You must provide a `points_field` when indexing points "
                    "that are not associated with a brain key"
                )

            points_field = self.key

        _generate_spatial_index(
            self.samples,
            self.points,
            points_field,
            self.sample_ids,
            label_ids=self.label_ids,
            patches_field=self.config.patches_field,
            create_index=create_index,
            progress=progress,
        )

        if self.key is not None:
            self.config.points_field = points_field
            self.save_config()

    def remove_index(self):
        """Removes the spatial index from these visualization results, if one
        exists.
        """
        points_field = self.config.points_field
        if points_field is None:
            return

        dataset = self.samples._root_dataset
        if self.config.patches_field is not None:
            _, points_field = dataset._get_label_field_path(
                self.config.patches_field, points_field
            )

        dataset.delete_sample_field(points_field, error_level=1)

        if self.key is not None:
            self.config.points_field = None
            self.save_config()

    @property
    def supports_auto_updates(self):
        """Whether this index can be incrementally updated by calling
        :meth:`add_samples` without manually providing `embeddings` or a
        `model`.
        """

        # Some visualization methods do not support incremental updates
        if not getattr(self.config.build(), "SUPPORTS_UPDATES", False):
            return False

        # The string name of a zoo model is required in order to call
        # `add_samples()` without manually providing `embeddings` or a `model`
        if self.config.model is None:
            return False

        # UMAP requires the embeddings to be stored on a field of the dataset
        # in order to rehydrate its reducer
        if (
            self.config.method == "umap"
            and self.config.embeddings_field is None
        ):
            return False

        # If no reducer/blob is stored, this is an older visualization that
        # cannot be incrementally updated
        if self._reducer is None and self._reducer_blob is None:
            return False

        return True

    def add_samples(
        self,
        samples,
        embeddings=None,
        model=None,
        model_kwargs=None,
        force_square=False,
        alpha=None,
        batch_size=None,
        num_workers=None,
        skip_failures=True,
        progress=None,
        skip_existing=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
    ):
        """Incrementally extends this visualization with the new samples in
        ``samples``.

        Embeddings for the new samples are extracted (or accepted) and
        transformed through the previously fitted reducer, then appended to
        the index. This is significantly cheaper than recomputing the entire
        visualization for datasets where only a few samples have been added.

        By default, samples that are already in the visualization are skipped,
        so it is safe to pass an entire dataset/view that contains both known
        and new samples.

        Only methods whose reducer supports ``transform()`` are supported
        (currently UMAP and PCA). For UMAP, the original
        ``config.embeddings_field`` must be set so that the kNN search
        structure can be rebuilt on demand from the dataset.

        The embedding source for the new samples is resolved in this order:

        1. ``embeddings`` if provided (array, dict, or field name)
        2. ``model`` if provided
        3. ``config.embeddings_field`` if populated on the new samples
        4. ``config.model`` if the zoo model name recorded on the original run
        5. otherwise, an error is raised

        Note that if the original :func:`compute_visualization` call was made
        with a :class:`fiftyone.core.models.Model` instance rather than the
        string name of a zoo model, ``config.model`` is ``None`` and you must
        pass ``model`` or ``embeddings`` explicitly here to extend the
        visualization.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
                containing samples to add to the visualization
            embeddings (None): pre-computed embeddings for the new samples.
                Can be a ``num_samples x num_dims`` array, a dict mapping
                sample/label IDs to embeddings, or the name of a field from
                which to load them. If the resolved source differs from
                ``config.embeddings_field``, the new embeddings are also
                written through to ``config.embeddings_field`` so that future
                UMAP reloads can rehydrate consistently
            model (None): a model to use to compute embeddings. If not
                provided, ``config.model`` is used if available
            model_kwargs (None): a dictionary of optional keyword arguments to
                pass to the model's ``Config`` when a model name is provided
            force_square (False): see :func:`compute_visualization`
            alpha (None): see :func:`compute_visualization`
            batch_size (None): see :func:`compute_visualization`
            num_workers (None): see :func:`compute_visualization`
            skip_failures (True): see :func:`compute_visualization`
            progress (None): see :func:`compute_visualization`
            skip_existing (True): if True, samples already in the
                visualization are silently dropped from ``samples`` before
                computing embeddings, so only new samples are projected
                through the reducer
            allow_existing (True): if False, raise an error when
                ``skip_existing`` is False and and a provided sample contains
                an ID that already exists in the index
            warn_existing (False): whether to log a warning if a point is not
                added because its ID already exists in the index
            reload (True): whether to refresh the current view after the
                update
        """
        fov.validate_collection(samples)
        if samples._root_dataset is not self._samples._root_dataset:
            raise ValueError(
                "You can only add samples from the same dataset to an "
                "existing visualization"
            )

        self._prepare_reducer(samples)

        config = self.config
        patches_field = config.patches_field
        canonical_field = config.embeddings_field
        source_descriptor = _describe_embeddings_source(embeddings)

        override_field = None
        if isinstance(embeddings, str):
            if embeddings != canonical_field:
                override_field = embeddings
            embeddings = None

        if isinstance(embeddings, np.ndarray):
            embeddings = _array_to_id_dict(
                samples, embeddings, patches_field=patches_field
            )

        if skip_existing:
            samples, embeddings = self._filter_known(samples, embeddings)
            if len(samples) == 0:
                logger.info("No new samples to add")
                if reload:
                    self.use_view(self._curr_view or self._samples)

                return

        effective_model = model
        effective_model_kwargs = model_kwargs

        source_field = override_field or canonical_field

        field_populated = _field_has_populated_values(
            samples, source_field, patches_field
        )

        if (
            embeddings is None
            and effective_model is None
            and not field_populated
        ):
            if config.model is not None:
                effective_model = config.model
                if effective_model_kwargs is None:
                    effective_model_kwargs = config.model_kwargs
            else:
                raise ValueError(
                    "No embeddings, model, or embeddings field are available "
                    "for the new samples. You must pass `model=` or "
                    "`embeddings=` explicitly to add_samples()"
                )

        forced_model_fallback = (
            effective_model is not None
            and source_field is not None
            and not field_populated
        )
        extraction_field = None if forced_model_fallback else source_field

        new_emb, new_sample_ids, new_label_ids = fbu.get_embeddings(
            samples,
            model=effective_model,
            model_kwargs=effective_model_kwargs,
            patches_field=patches_field,
            embeddings_field=extraction_field,
            embeddings=embeddings,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=progress,
        )

        expected_dim = _expected_input_dim(self._reducer)
        if expected_dim is not None and new_emb.shape[1] != expected_dim:
            raise ValueError(
                "embeddings have dimension %d but the fitted reducer "
                "expects dimension %d" % (new_emb.shape[1], expected_dim)
            )

        new_sample_ids_out, new_label_ids_out, ii, jj = fbu.add_ids(
            new_sample_ids,
            new_label_ids,
            self.sample_ids,
            self.label_ids,
            patches_field=patches_field,
            overwrite=skip_existing,
            allow_existing=allow_existing,
            warn_existing=warn_existing,
        )

        if ii.size == 0:
            if reload:
                self.use_view(self._curr_view or self._samples)

            return

        if self.config.method == "umap":
            n_new = int(ii.size)
            n_train = self._reducer.embedding_.shape[0]
            ratio = n_new / n_train
            if ratio > 0.2:
                logger.warning(
                    "Projecting %d new samples through a fitted UMAP that "
                    "was trained on %d samples (%.0f%%). This is an "
                    "approximate transform — the manifold is not refit. "
                    "For batches this large, consider recomputing the "
                    "visualization via compute_visualization() for a more "
                    "faithful embedding",
                    n_new,
                    n_train,
                    100 * ratio,
                )
            else:
                logger.warning(
                    "Projecting %d new samples through a fitted UMAP that "
                    "was trained on %d samples (%.0f%%). This is an "
                    "approximate transform and not equivalent to a full "
                    "refit. Quality depends on the new samples being "
                    "in-distribution with the training set",
                    n_new,
                    n_train,
                    100 * ratio,
                )

        try:
            new_points = np.asarray(self._reducer.transform(new_emb[ii, :]))
        except Exception as e:
            # same environment-mismatch failure mode as reducer hydration;
            # see _prepare_reducer()
            raise ValueError(_REDUCER_APPLY_ERROR) from e

        n = len(self.points)
        m = int(jj.max()) - n + 1
        if m > 0:
            self.points = np.concatenate(
                (
                    self.points,
                    np.empty(
                        (m, self.points.shape[1]), dtype=self.points.dtype
                    ),
                )
            )

        self.points[jj] = new_points
        self.sample_ids = new_sample_ids_out
        if patches_field is not None:
            self.label_ids = new_label_ids_out

        if canonical_field is not None:
            needs_write_through = (
                source_descriptor in ("<dict>", "<ndarray>")
                or (
                    override_field is not None
                    and override_field != canonical_field
                )
                or forced_model_fallback
            )
            if needs_write_through:
                if forced_model_fallback:
                    logger.info(
                        "Computed %d new embeddings via model %r and stored "
                        "them in embeddings_field=%r",
                        int(ii.size),
                        effective_model,
                        canonical_field,
                    )
                else:
                    logger.warning(
                        "Embeddings for the new samples were sourced from "
                        "%r but will also be stored in this run's "
                        "embeddings_field=%r so that UMAP rehydration on "
                        "future loads remains consistent",
                        source_descriptor,
                        canonical_field,
                    )

                fbu.add_embeddings(
                    self._samples,
                    new_emb[ii],
                    new_sample_ids[ii],
                    new_label_ids[ii] if new_label_ids is not None else None,
                    canonical_field,
                    patches_field=patches_field,
                )

        if config.points_field is not None:
            self._write_points_field(
                new_points,
                new_sample_ids[ii],
                new_label_ids[ii] if new_label_ids is not None else None,
            )

        if self.key is not None:
            self.save()

        if reload:
            self.use_view(self._curr_view or self._samples)

    def _prepare_reducer(self, samples):
        if not getattr(self.config.build(), "SUPPORTS_UPDATES", False):
            raise ValueError(
                "method '%s' does not support incremental updates; please "
                "recompute via compute_visualization()" % self.config.method
            )

        if (
            self.config.method == "umap"
            and self.config.embeddings_field is None
        ):
            raise ValueError(
                "UMAP incremental updates require an embeddings field. "
                "Please recompute via "
                "compute_visualization(..., embeddings=<field name>) to "
                "enable incremental updates"
            )

        if self._reducer is None:
            if self._reducer_blob is None:
                raise ValueError(
                    "This visualization has no stored reducer. It may have "
                    "been computed before incremental updates were supported. "
                    "Please recompute via compute_visualization() to enable "
                    "incremental updates"
                )

            self._reducer = _unpickle_reducer(self._reducer_blob)

        if self.config.method == "umap":
            if getattr(self._reducer, "_raw_data", None) is None:
                training_embeddings = self._load_training_embeddings()
                try:
                    _hydrate_umap_reducer(self._reducer, training_embeddings)
                except Exception as e:
                    # a reducer pickled under a different Python/umap/numba
                    # unpickles cleanly but fails here, deep in numba (e.g.
                    # ``AssertionError: key already in dictionary``), so
                    # translate anything into an actionable error
                    raise ValueError(_REDUCER_APPLY_ERROR) from e

    def _load_training_embeddings(self):
        n_train = self._reducer.embedding_.shape[0]

        patches_field = self.config.patches_field
        if patches_field is not None:
            training_ids = self.label_ids[:n_train]
        else:
            training_ids = self.sample_ids[:n_train]

        training_view = (
            self._samples.select(list(training_ids), ordered=True)
            if patches_field is None
            else self._samples.select_labels(
                ids=list(training_ids), fields=patches_field
            )
        )

        embeddings, sample_ids, label_ids = fbu.get_embeddings(
            training_view,
            patches_field=patches_field,
            embeddings_field=self.config.embeddings_field,
        )

        ids_returned = label_ids if patches_field is not None else sample_ids

        returned_set = set(
            ids_returned.tolist()
            if hasattr(ids_returned, "tolist")
            else ids_returned
        )
        expected_set = set(training_ids.tolist())
        if expected_set != returned_set:
            missing = sorted(expected_set - returned_set)
            raise ValueError(
                "Cannot rehydrate UMAP reducer: %d training %s are missing "
                "from embeddings_field=%r (e.g. %s). The original training "
                "embeddings are not recoverable; please recompute the "
                "visualization"
                % (
                    len(missing),
                    "labels" if patches_field is not None else "samples",
                    self.config.embeddings_field,
                    missing[:5],
                )
            )

        order = {_id: i for i, _id in enumerate(ids_returned)}
        idx = np.array([order[_id] for _id in training_ids])
        return embeddings[idx]

    def _filter_known(self, samples, embeddings):
        patches_field = self.config.patches_field

        new_ids, num_total = self._diff_ids(samples)

        if patches_field is not None and self.label_ids is not None:
            if len(new_ids) < num_total:
                if new_ids:
                    samples = samples.select_labels(
                        ids=new_ids, fields=patches_field
                    )
                else:
                    samples = samples.limit(0)
        else:
            if len(new_ids) != num_total:
                samples = samples.select(new_ids)

        if isinstance(embeddings, dict):
            if patches_field is not None and self.label_ids is not None:
                known_keys = set(self.label_ids.tolist())
            else:
                known_keys = set(self.sample_ids.tolist())
            embeddings = {
                _id: vec
                for _id, vec in embeddings.items()
                if _id not in known_keys
            }

        return samples, embeddings

    def _diff_ids(self, samples):
        patches_field = self.config.patches_field

        if patches_field is not None and self.label_ids is not None:
            _, label_ids_path = samples._get_label_field_path(
                patches_field, "id"
            )
            all_ids = samples.values(label_ids_path, unwind=True)
            known = set(self.label_ids.tolist())
        else:
            all_ids = samples.values("id")
            known = set(self.sample_ids.tolist())

        new_ids = [_id for _id in all_ids if _id not in known]
        return new_ids, len(all_ids)

    def get_new_ids(self, samples=None, include_training_size=True):
        """Returns the IDs of the samples/patches in the given collection
        that are not present in this visualization, along with the number of
        embeddings that were used to fit this visualization's reducer.

        For sample-level visualizations, sample IDs are returned; for
        patch-level visualizations, label IDs are returned.

        The training size can be used to decide whether an incremental
        update via :meth:`add_samples` is appropriate; when the number of
        new IDs is large relative to the training size (e.g. > 20%),
        recomputing the visualization will produce a more faithful embedding.

        Computing the training size requires unpickling the stored reducer,
        which for UMAP imports the ``umap`` package and can take several
        seconds on the first use in a process. Pass
        ``include_training_size=False`` to skip that work when only the IDs
        are needed (e.g. to render a count in an interactive context).

        Args:
            samples (None): a
                :class:`fiftyone.core.collections.SampleCollection` to check.
                By default, the full collection on which this run was
                performed is used
            include_training_size (True): whether to compute the training
                size, which loads the stored reducer. When ``False``, the
                training size is returned as ``None``

        Returns:
            a tuple of

            -   a list of new IDs
            -   the number of embeddings used to fit the reducer, or ``None``
                if it was not computed or no reducer is available for this
                run
        """
        if samples is None:
            samples = self._samples

        new_ids, _ = self._diff_ids(samples)

        if not include_training_size:
            return new_ids, None

        return new_ids, self._training_size()

    def needs_update(self, samples=None):
        """Determines whether the given collection contains samples/patches
        that are not present in this visualization.

        Use :meth:`add_samples` to add the new samples to the visualization.

        Args:
            samples (None): a
                :class:`fiftyone.core.collections.SampleCollection` to check.
                By default, the full collection on which this run was
                performed is used

        Returns:
            True/False
        """
        if samples is None:
            samples = self._samples

        new_ids, _ = self._diff_ids(samples)
        return len(new_ids) > 0

    def _training_size(self):
        if self._reducer is None and self._reducer_blob is not None:
            try:
                self._reducer = _unpickle_reducer(self._reducer_blob)
            except RuntimeError:
                return None

        reducer = self._reducer
        if reducer is None:
            return None

        # UMAP
        embedding = getattr(reducer, "embedding_", None)
        if embedding is not None:
            return embedding.shape[0]

        # PCA
        n_samples = getattr(reducer, "n_samples_", None)
        if n_samples is not None:
            return int(n_samples)

        return None

    def _write_points_field(self, new_points, sample_ids, label_ids):
        config = self.config
        points_field = config.points_field
        patches_field = config.patches_field

        dataset = self._samples._root_dataset
        if patches_field is not None:
            _, points_field_path = dataset._get_label_field_path(
                patches_field, points_field
            )
            values = dict(zip(label_ids, new_points.astype(float).tolist()))
            dataset.set_label_values(points_field_path, values)
        else:
            values = dict(zip(sample_ids, new_points.astype(float).tolist()))
            dataset.set_values(points_field, values, key_field="id")

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        points = _parse_serialized_array(d["points"])
        sample_ids = _parse_serialized_array(d.get("sample_ids", None))
        label_ids = _parse_serialized_array(d.get("label_ids", None))

        reducer_blob = d.get("reducer_blob", None)

        return cls(
            samples,
            config,
            brain_key,
            points,
            sample_ids=sample_ids,
            label_ids=label_ids,
            reducer_blob=reducer_blob,
        )


class VisualizationConfig(fob.BrainMethodConfig):
    """Base class for configuring visualization methods.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        points_field (None): the name of a field in which to store the
            visualization points, if requested
        similarity_index (None): the similarity index containing the
            embeddings, if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        num_dims (2): the dimension of the visualization space
    """

    def __init__(
        self,
        embeddings_field=None,
        points_field=None,
        similarity_index=None,
        model=None,
        model_kwargs=None,
        patches_field=None,
        num_dims=2,
        **kwargs,
    ):
        if similarity_index is not None and not etau.is_str(similarity_index):
            similarity_index = similarity_index.key

        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.points_field = points_field
        self.similarity_index = similarity_index
        self.model = model
        self.model_kwargs = model_kwargs
        self.patches_field = patches_field
        self.num_dims = num_dims
        super().__init__(**kwargs)

    @property
    def type(self):
        return "visualization"


class Visualization(fob.BrainMethod):
    SUPPORTS_UPDATES = False

    def fit_reducer(self, embeddings):
        raise NotImplementedError("subclass must implement fit_reducer()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)
        elif self.config.points_field is not None:
            fields.append(self.config.points_field)

        return fields

    def rename(self, samples, key, new_key):
        patches_field = self.config.patches_field
        points_field = self.config.points_field
        dataset = samples._root_dataset

        if points_field is not None and points_field == key:
            old_path = key
            new_path = new_key
            if patches_field is not None:
                _, old_path = dataset._get_label_field_path(
                    patches_field, old_path
                )
                _, new_path = dataset._get_label_field_path(
                    patches_field, new_path
                )

            self.config.points_field = new_key
            self.update_run_config(samples, key, self.config)

            dataset.rename_sample_field(old_path, new_path)

    def cleanup(self, samples, key):
        patches_field = self.config.patches_field
        points_field = self.config.points_field
        dataset = samples._root_dataset

        if points_field is not None:
            if patches_field is not None:
                _, points_field = dataset._get_label_field_path(
                    patches_field, points_field
                )

            dataset.delete_sample_field(points_field, error_level=1)


class UMAPVisualizationConfig(VisualizationConfig):
    """Configuration for Uniform Manifold Approximation and Projection (UMAP)
    embedding visualization.

    See https://github.com/lmcinnes/umap for more information about the
    supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        points_field (None): the name of a field in which to store the
            visualization points, if requested
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
        points_field=None,
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
            points_field=points_field,
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
    SUPPORTS_UPDATES = True

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

    def fit_reducer(self, embeddings):
        _umap = umap.UMAP(
            n_components=self.config.num_dims,
            n_neighbors=self.config.num_neighbors,
            metric=self.config.metric,
            min_dist=self.config.min_dist,
            random_state=self.config.seed,
            verbose=self.config.verbose,
        )
        points = _umap.fit_transform(embeddings)
        _strip_umap_training_data(_umap)
        return points, _umap


class TSNEVisualizationConfig(VisualizationConfig):
    """Configuration for t-distributed Stochastic Neighbor Embedding (t-SNE)
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    for more information about the supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        points_field (None): the name of a field in which to store the
            visualization points, if requested
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
        points_field=None,
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
            points_field=points_field,
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
    def fit_reducer(self, embeddings):
        if self.config.pca_dims is not None:
            _pca = skd.PCA(
                n_components=self.config.pca_dims,
                svd_solver=self.config.svd_solver,
                random_state=self.config.seed,
            )
            embeddings = _pca.fit_transform(embeddings)

        embeddings = embeddings.astype(np.float32, copy=False)

        verbose = 2 if self.config.verbose else 0

        sklearn_version = version.parse(sklearn.__version__)
        iter_param = (
            "max_iter"
            if sklearn_version >= version.parse("1.5.0")
            else "n_iter"
        )

        _tsne = skm.TSNE(
            n_components=self.config.num_dims,
            perplexity=self.config.perplexity,
            learning_rate=self.config.learning_rate,
            metric=self.config.metric,
            init="pca",
            random_state=self.config.seed,
            verbose=verbose,
            **{iter_param: self.config.max_iters},
        )
        points = _tsne.fit_transform(embeddings)

        return points, None


class PCAVisualizationConfig(VisualizationConfig):
    """Configuration for principal component analysis (PCA) embedding
    visualization.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    for more information about the supported parameters.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        points_field (None): the name of a field in which to store the
            visualization points, if requested
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
        points_field=None,
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
            points_field=points_field,
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
    SUPPORTS_UPDATES = True

    def fit_reducer(self, embeddings):
        _pca = skd.PCA(
            n_components=self.config.num_dims,
            svd_solver=self.config.svd_solver,
            random_state=self.config.seed,
        )
        points = _pca.fit_transform(embeddings)
        return points, _pca


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
    def fit_reducer(self, embeddings):
        raise NotImplementedError(
            "The low-dimensional representation must be manually provided "
            "when using this method"
        )


def _parse_serialized_array(value):
    """Decodes an array-valued field of serialized visualization results.

    Two encodings exist: current results store arrays as compressed
    ``.npy`` strings (see :meth:`VisualizationResults.serialize`), while
    results written by earlier versions store plain JSON lists.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return fou.deserialize_numpy_array(value, ascii=True)

    return np.array(value)


def _field_has_populated_values(samples, embeddings_field, patches_field=None):
    if embeddings_field is None:
        return False

    if not fbu._has_embeddings_field(samples, embeddings_field, patches_field):
        return False

    try:
        if patches_field is not None:
            filtered = samples.filter_labels(
                patches_field, foe.ViewField(embeddings_field) != None
            )
            _, ids_path = samples._get_label_field_path(patches_field, "id")
            return len(filtered.values(ids_path, unwind=True)) > 0

        return len(samples.exists(embeddings_field)) > 0
    except Exception:
        return False


def _describe_embeddings_source(embeddings):
    if isinstance(embeddings, str):
        return embeddings

    if isinstance(embeddings, dict):
        return "<dict>"

    if isinstance(embeddings, np.ndarray):
        return "<ndarray>"

    return None


def _array_to_id_dict(samples, embeddings, patches_field=None):
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings array must be 2D; got shape %s" % (embeddings.shape,)
        )

    sample_ids, label_ids = fbu.get_ids(
        samples,
        patches_field=patches_field,
        data=embeddings,
        data_type="embeddings",
    )

    ids = label_ids if patches_field is not None else sample_ids
    if len(ids) != len(embeddings):
        raise ValueError(
            "embeddings array length (%d) does not match the number of "
            "samples (%d)" % (len(embeddings), len(ids))
        )

    return {_id: vec for _id, vec in zip(ids, embeddings)}


def _expected_input_dim(reducer):
    n_features = getattr(reducer, "n_features_in_", None)
    if n_features is not None:
        return n_features

    raw = getattr(reducer, "_raw_data", None)
    if raw is not None:
        return raw.shape[1]

    return None


def _strip_umap_training_data(reducer):
    reducer._raw_data = None
    if getattr(reducer, "_knn_search_index", None) is not None:
        idx = reducer._knn_search_index
        reducer._knn_index_params = {
            "n_trees": idx.n_trees,
            "n_iters": idx.n_iters,
            "max_candidates": idx.max_candidates,
        }
        reducer._knn_search_index = None

    # The jitted distance functions serialize as raw Python bytecode, which
    # is not stable across Python versions: unpickling on a different
    # version succeeds but numba's JIT crashes at transform time. They
    # carry no fitted state and are fully determined by the (persisted)
    # metric strings, so strip them here and rebind from the local umap
    # installation on load
    reducer._input_distance_func = None
    reducer._output_distance_func = None
    reducer._inverse_distance_func = None


def _rebind_umap_distance_funcs(reducer):
    import umap.distances as dist

    metric = reducer.metric
    if not etau.is_str(metric) or metric not in dist.named_distances:
        raise RuntimeError(
            "Cannot rebind distance functions for metric %r; please "
            "recompute the embeddings visualization" % (metric,)
        )

    reducer._input_distance_func = dist.named_distances[metric]
    reducer._inverse_distance_func = dist.named_distances_with_gradients.get(
        metric, None
    )
    reducer._output_distance_func = dist.named_distances_with_gradients[
        reducer.output_metric
    ]


def _hydrate_umap_reducer(reducer, embeddings):
    reducer._raw_data = embeddings

    _rebind_umap_distance_funcs(reducer)

    if getattr(reducer, "_small_data", True):
        return

    # Need to reconstruct the index since we are not storing the search index by
    # in the brain result as this would involve storing the embeddings vectors in
    # the brain result

    from pynndescent import NNDescent
    from sklearn.utils import check_random_state

    params = reducer._knn_index_params

    reducer._knn_search_index = NNDescent(
        embeddings,
        n_neighbors=reducer._n_neighbors,
        metric=reducer.metric,
        metric_kwds=reducer._metric_kwds,
        random_state=check_random_state(reducer.random_state),
        n_trees=params["n_trees"],
        n_iters=params["n_iters"],
        max_candidates=params["max_candidates"],
        low_memory=reducer.low_memory,
        n_jobs=reducer.n_jobs,
        verbose=reducer.verbose,
        compressed=False,
    )


def _pickle_reducer(reducer):
    if reducer is None:
        return None

    blob = zlib.compress(pickle.dumps(reducer))
    return base64.b64encode(blob).decode("ascii")


def _unpickle_reducer(blob):
    if blob is None:
        return None

    try:
        return pickle.loads(zlib.decompress(base64.b64decode(blob)))
    except Exception as e:
        raise RuntimeError(
            "Failed to deserialize fitted reducer (likely a UMAP/sklearn "
            "version mismatch); please recompute with compute_visualization()"
        ) from e
