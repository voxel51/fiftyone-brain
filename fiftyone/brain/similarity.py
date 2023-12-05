"""
Similarity interface.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
from copy import deepcopy
import inspect
import logging

from bson import ObjectId
import numpy as np

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.brain as fob
import fiftyone.core.context as foc
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.patches as fop
import fiftyone.core.stages as fos
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz
from fiftyone import ViewField as F

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mobilenet-v2-imagenet-torch"
_DEFAULT_BATCH_SIZE = None


def compute_similarity(
    samples,
    patches_field,
    embeddings,
    brain_key,
    model,
    force_square,
    alpha,
    batch_size,
    num_workers,
    skip_failures,
    backend,
    **kwargs,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    # Allow for `embeddings_field=XXX` and `embeddings=False` together
    embeddings_field = kwargs.pop("embeddings_field", None)
    if embeddings_field is not None or etau.is_str(embeddings):
        if embeddings_field is None:
            embeddings_field = embeddings
            embeddings = None

        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings_field,
            patches_field=patches_field,
        )
    else:
        embeddings_field = None
        embeddings_exist = None

    if model is None and embeddings is None and not embeddings_exist:
        model = _DEFAULT_MODEL
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    if etau.is_str(model):
        _model = foz.load_zoo_model(model)
        try:
            supports_prompts = _model.can_embed_prompts
        except:
            supports_prompts = None
    else:
        _model = model
        supports_prompts = None

    config = _parse_config(
        backend,
        embeddings_field=embeddings_field,
        patches_field=patches_field,
        model=model,
        supports_prompts=supports_prompts,
        **kwargs,
    )
    brain_method = config.build()
    brain_method.ensure_requirements()

    if brain_key is not None:
        # Don't allow overwriting an existing run with same key, since we
        # need the existing run in order to perform workflows like
        # automatically cleaning up the backend's index
        brain_method.register_run(samples, brain_key, overwrite=False)

    results = brain_method.initialize(samples, brain_key)

    get_embeddings = embeddings is not False
    if not results.is_external and results.total_index_size > 0:
        # No need to load embeddings because the index already has them
        get_embeddings = False

    if get_embeddings:
        # Don't immediatly store embeddings in DB; let `add_to_index()` do it
        if not embeddings_exist:
            embeddings_field = None

        embeddings, sample_ids, label_ids = fbu.get_embeddings(
            samples,
            model=_model,
            patches_field=patches_field,
            embeddings=embeddings,
            embeddings_field=embeddings_field,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
        )
    else:
        embeddings = None

    if embeddings is not None:
        results.add_to_index(embeddings, sample_ids, label_ids=label_ids)

    brain_method.save_run_results(samples, brain_key, results)

    return results


def _parse_config(name, **kwargs):
    if name is None:
        name = fb.brain_config.default_similarity_backend

    if inspect.isclass(name):
        return name(**kwargs)

    backends = fb.brain_config.similarity_backends

    if name not in backends:
        raise ValueError(
            "Unsupported backend '%s'. The available backends are %s"
            % (name, sorted(backends.keys()))
        )

    params = deepcopy(backends[name])

    config_cls = kwargs.pop("config_cls", None)

    if config_cls is None:
        config_cls = params.pop("config_cls", None)

    if config_cls is None:
        raise ValueError("Similarity backend '%s' has no `config_cls`" % name)

    if etau.is_str(config_cls):
        config_cls = etau.get_class(config_cls)

    params.update(**kwargs)
    return config_cls(**params)


class SimilarityConfig(fob.BrainMethodConfig):
    """Similarity configuration.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (False): whether this run supports prompt queries
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = None

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.supports_prompts = supports_prompts
        super().__init__(**kwargs)

    @property
    def type(self):
        return "similarity"

    @property
    def method(self):
        """The name of the similarity backend."""
        raise NotImplementedError("subclass must implement method")

    @property
    def max_k(self):
        """A maximum k value for nearest neighbor queries, or None if there is
        no limit.
        """
        raise NotImplementedError("subclass must implement max_k")

    @property
    def supports_least_similarity(self):
        """Whether this backend supports least similarity queries."""
        raise NotImplementedError(
            "subclass must implement supports_least_similarity"
        )

    @property
    def supported_aggregations(self):
        """A tuple of supported values for the ``aggregation`` parameter of the
        backend's
        :meth:`sort_by_similarity() <SimilarityIndex.sort_by_similarity>` and
        :meth:`_kneighbors() <SimilarityIndex._kneighbors>` methods.
        """
        raise NotImplementedError(
            "subclass must implement supported_aggregations"
        )

    def load_credentials(self, **kwargs):
        self._load_parameters(**kwargs)

    def _load_parameters(self, **kwargs):
        name = self.method
        parameters = fb.brain_config.similarity_backends.get(name, {})

        for name, value in kwargs.items():
            if value is None:
                value = parameters.get(name, None)

            if value is not None:
                setattr(self, name, value)


class Similarity(fob.BrainMethod):
    """Base class for similarity factories.

    Args:
        config: a :class:`SimilarityConfig`
    """

    def initialize(self, samples, brain_key):
        """Initializes a similarity index.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleColllection`
            brain_key: the brain key

        Returns:
            a :class:`SimilarityIndex`
        """
        raise NotImplementedError("subclass must implement initialize()")

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields


class SimilarityIndex(fob.BrainResults):
    """Base class for similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`Similarity` backend
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)

        self._model = None
        self._curr_view = None
        self._curr_sample_ids = None
        self._curr_label_ids = None
        self._curr_keep_inds = None
        self._curr_missing_size = None
        self._last_view = None
        self._last_views = []

        self.use_view(samples)

    def __enter__(self):
        self._last_views.append(self._last_view)
        return self

    def __exit__(self, *args):
        try:
            last_view = self._last_views.pop()
        except:
            last_view = self._samples

        self.use_view(last_view)

    @property
    def config(self):
        """The :class:`SimilarityConfig` for these results."""
        return self._config

    @property
    def is_external(self):
        """Whether this similarity index manages its own embeddings (True) or
        loads them directly from the ``embeddings_field`` of the dataset
        (False).
        """
        return True  # assume external unless explicitly overridden

    @property
    def sample_ids(self):
        """The sample IDs of the full index, or ``None`` if not supported."""
        return None

    @property
    def label_ids(self):
        """The label IDs of the full index, or ``None`` if not applicable or
        not supported.
        """
        return None

    @property
    def total_index_size(self):
        """The total number of data points in the index.

        If :meth:`use_view` has been called to restrict the index, this value
        may be larger than the current :meth:`index_size`.
        """
        raise NotImplementedError("subclass must implement total_index_size")

    @property
    def has_view(self):
        """Whether the index is currently restricted to a view.

        Use :meth:`use_view` to restrict the index to a view, and use
        :meth:`clear_view` to reset to the full index.
        """
        return self._curr_view.view() != self._samples.view()

    @property
    def view(self):
        """The :class:`fiftyone.core.collections.SampleCollection` against
        which results are currently being generated.

        If :meth:`use_view` has been called, this view may be different than
        the collection on which the full index was generated.
        """
        return self._curr_view

    @property
    def current_sample_ids(self):
        """The sample IDs of the currently active data points in the index.

        If :meth:`use_view` has been called, this may be a subset of the full
        index.
        """
        return self._curr_sample_ids

    @property
    def current_label_ids(self):
        """The label IDs of the currently active data points in the index, or
        ``None`` if not applicable.

        If :meth:`use_view` has been called, this may be a subset of the full
        index.
        """
        return self._curr_label_ids

    @property
    def _current_inds(self):
        """The indices of :meth:`current_sample_ids` in :meth:`sample_ids`, or
        ``None`` if not supported or if the full index is currently being used.
        """
        return self._curr_keep_inds

    @property
    def index_size(self):
        """The number of active data points in the index.

        If :meth:`use_view` has been called to restrict the index, this
        property will reflect the size of the active index.
        """
        return len(self._curr_sample_ids)

    @property
    def missing_size(self):
        """The total number of data points in :meth:`view` that are missing
        from this index, or ``None`` if unknown.

        This property is only applicable when :meth:`use_view` has been called,
        and it will be ``None`` if no data points are missing or when the
        backend does not support it.
        """
        return self._curr_missing_size

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
    ):
        """Adds the given embeddings to the index.

        Args:
            embeddings: a ``num_embeddings x num_dims`` array of embeddings
            sample_ids: a ``num_embeddings`` array of sample IDs
            label_ids (None): a ``num_embeddings`` array of label IDs, if
                applicable
            overwrite (True): whether to replace (True) or ignore (False)
                existing embeddings with the same sample/label IDs
            allow_existing (True): whether to ignore (True) or raise an error
                (False) when ``overwrite`` is False and a provided ID already
                exists in the
            warn_missing (False): whether to log a warning if an embedding is
                not added to the index because its ID already exists
            reload (True): whether to call :meth:`reload` to refresh the
                current view after the update
        """
        raise NotImplementedError("subclass must implement add_to_index()")

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        """Removes the specified embeddings from the index.

        Args:
            sample_ids (None): an array of sample IDs
            label_ids (None): an array of label IDs, if applicable
            allow_missing (True): whether to allow the index to not contain IDs
                that you provide (True) or whether to raise an error in this
                case (False)
            warn_missing (False): whether to log a warning if the index does
                not contain IDs that you provide
            reload (True): whether to call :meth:`reload` to refresh the
                current view after the update
        """
        raise NotImplementedError(
            "subclass must implement remove_from_index()"
        )

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        """Retrieves the embeddings for the given IDs from the index.

        If no IDs are provided, the entire index is returned.

        Args:
            sample_ids (None): a sample ID or list of sample IDs for which to
                retrieve embeddings
            label_ids (None): a label ID or list of label IDs for which to
                retrieve embeddings
            allow_missing (True): whether to allow the index to not contain IDs
                that you provide (True) or whether to raise an error in this
                case (False)
            warn_missing (False): whether to log a warning if the index does
                not contain IDs that you provide

        Returns:
            a tuple of:

            -   a ``num_embeddings x num_dims`` array of embeddings
            -   a ``num_embeddings`` array of sample IDs
            -   a ``num_embeddings`` array of label IDs, if applicable, or else
                ``None``
        """
        raise NotImplementedError("subclass must implement get_embeddings()")

    def use_view(self, samples, allow_missing=True, warn_missing=False):
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

            results = fob.compute_similarity(dataset)
            print(results.index_size)  # 200

            view = dataset.take(50)

            with results.use_view(view):
                print(results.index_size)  # 50

                results.find_unique(10)
                print(results.unique_ids)

                plot = results.visualize_unique()
                plot.show()

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
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
            samples,
            self.sample_ids,
            self.label_ids,
            patches_field=self.config.patches_field,
            allow_missing=allow_missing,
            warn_missing=warn_missing,
        )

        if good_inds is not None:
            missing_size = good_inds.size - np.count_nonzero(good_inds)
        else:
            missing_size = None

        self._last_view = self._curr_view
        self._curr_view = samples
        self._curr_sample_ids = sample_ids
        self._curr_label_ids = label_ids
        self._curr_keep_inds = keep_inds
        self._curr_missing_size = missing_size

        return self

    def clear_view(self):
        """Clears the view set by :meth:`use_view`, if any.

        Subsequent operations will be performed on the full index.
        """
        self.use_view(self._samples)

    def reload(self):
        """Reloads the index for the current view.

        Subclasses may override this method, but by default this method simply
        passes the current :meth:`view` back into :meth:`use_view`, which
        updates the index's current ID set based on any changes to the view
        since the index was last loaded.
        """
        self.use_view(self._curr_view)

    def cleanup(self):
        """Deletes the similarity index from the backend."""
        raise NotImplementedError("subclass must implement cleanup()")

    def values(self, path_or_expr):
        """Extracts a flat list of values from the given field or expression
        corresponding to the current :meth:`view`.

        This method always returns values in the same order as
        :meth:`current_sample_ids` and :meth:`current_label_ids`.

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
        samples = self.view
        patches_field = self.config.patches_field

        if patches_field is not None:
            ids = self.current_label_ids
        else:
            ids = self.current_sample_ids

        return fbu.get_values(
            samples, path_or_expr, ids, patches_field=patches_field
        )

    def sort_by_similarity(
        self,
        query,
        k=None,
        reverse=False,
        aggregation="mean",
        dist_field=None,
        _mongo=False,
    ):
        """Returns a view that sorts the samples/labels in :meth:`view` by
        similarity to the specified query.

        When querying by IDs, the query can be any ID(s) in the full index of
        this instance, even if the current :meth:`view` contains a subset of
        the full index.

        Args:
            query: the query, which can be any of the following:

                -   an ID or iterable of IDs
                -   a ``num_dims`` vector or ``num_queries x num_dims`` array
                    of vectors
                -   a prompt or iterable of prompts (if supported by the index)

            k (None): the number of matches to return. Some backends may
                support ``None``, in which case all samples will be sorted
            reverse (False): whether to sort by least similarity (True) or
                greatest similarity (False). Some backends may not support
                least similarity
            aggregation ("mean"): the aggregation method to use when multiple
                queries are provided. The default is ``"mean"``, which means
                that the query vectors are averaged prior to searching. Some
                backends may support additional options
            dist_field (None): the name of a float field in which to store the
                distance of each example to the specified query. The field is
                created if necessary

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        samples = self.view
        patches_field = self.config.patches_field

        selecting_samples = patches_field is None or isinstance(
            samples, fop.PatchesView
        )

        kwargs = dict(
            query=self._parse_query(query),
            k=k,
            reverse=reverse,
            aggregation=aggregation,
            return_dists=dist_field is not None,
        )

        if dist_field is not None:
            ids, dists = self._kneighbors(**kwargs)
        else:
            ids = self._kneighbors(**kwargs)

        if not selecting_samples:
            label_ids = ids

            _ids = set(ids)
            bools = np.array([_id in _ids for _id in self.current_label_ids])
            sample_ids = self.current_sample_ids[bools]

        # Store query distances
        if dist_field is not None:
            if selecting_samples:
                values = dict(zip(ids, dists))
                samples.set_values(dist_field, values, key_field="id")
            else:
                label_type, path = samples._get_label_field_path(
                    patches_field, dist_field
                )
                if issubclass(label_type, fol._LABEL_LIST_FIELDS):
                    samples._set_list_values_by_id(
                        path,
                        sample_ids,
                        label_ids,
                        dists,
                        path.rsplit(".", 1)[0],
                    )
                else:
                    values = dict(zip(sample_ids, dists))
                    samples.set_values(path, values, key_field="id")

        # Construct sorted view
        stages = []

        if selecting_samples:
            stage = fos.Select(ids, ordered=True)
            stages.append(stage)
        else:
            # Sorting by object similarity but this is not a patches view, so
            # arrange the samples in order of their first occuring label
            result_sample_ids = _unique_no_sort(sample_ids)
            stage = fos.Select(result_sample_ids, ordered=True)
            stages.append(stage)

            if k is not None:
                _ids = [ObjectId(_id) for _id in ids]
                stage = fos.FilterLabels(patches_field, F("_id").is_in(_ids))
                stages.append(stage)

        if _mongo:
            pipeline = []
            for stage in stages:
                stage.validate(samples)
                pipeline.extend(stage.to_mongo(samples))

            return pipeline

        view = samples
        for stage in stages:
            view = view.add_stage(stage)

        return view

    def _parse_query(self, query):
        if query is None:
            raise ValueError("At least one query must be provided")

        if isinstance(query, np.ndarray):
            # Query by vector(s)
            if query.size == 0:
                raise ValueError("At least one query vector must be provided")

            return query

        if etau.is_str(query):
            query = [query]
        else:
            query = list(query)

        if not query:
            raise ValueError("At least one query must be provided")

        if etau.is_numeric(query[0]):
            return np.asarray(query)

        try:
            ObjectId(query[0])
            is_prompts = False
        except:
            is_prompts = True

        if is_prompts:
            if not self.config.supports_prompts:
                raise ValueError(
                    "Invalid query '%s'; this model does not support prompts"
                    % query[0]
                )

            model = self.get_model()
            with model:
                return model.embed_prompts(query)

        return query

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        """Returns the k-nearest neighbors for the given query.

        This method should only return results from the current :meth:`view`.

        Args:
            query (None): the query, which can be any of the following:

                -   an ID or list of IDs for which to return neighbors
                -   an embedding or ``num_queries x num_dim`` array of
                    embeddings for which to return neighbors
                -   Some backends may also support ``None``, in which case the
                    neighbors for all points in the current :meth:`view are
                    returned

            k (None): the number of neighbors to return. Some backends may
                enforce upper bounds on this parameter
            reverse (False): whether to sort by least similarity (True) or
                greatest similarity (False). Some backends may not support
                least similarity
            aggregation (None): an optional aggregation method to use when
                multiple queries are provided. All backends must support
                ``"mean"``, which averages query vectors prior to searching.
                Backends may support additional options as well
            return_dists (False): whether to return query-neighbor distances

        Returns:
            the query result, in one of the following formats:

                -   an ``(ids, dists)`` tuple, when ``return_dists`` is True
                -   ``ids``, when ``return_dists`` is False

            In the above, ``ids`` contains the IDs of the nearest neighbors, in
            one of the following formats:

                -   a list of nearest neighbor IDs, when a single query ID or
                    vector is provided, **or** when an ``aggregation`` is
                    provided
                -   a list of lists of nearest neighbor IDs, when multiple
                    query IDs/vectors and no ``aggregation`` is provided
                -   a list of arrays of the **integer indexes** (not IDs) of
                    nearest neighbor points for every vector in the index, when
                    no query is provided

            and ``dists`` contains the corresponding query-neighbor distances
            for each result in ``ids``
        """
        raise NotImplementedError("subclass must implement _kneighbors()")

    def get_model(self):
        """Returns the stored model for this index.

        Returns:
            a :class:`fiftyone.core.models.Model`
        """
        if self._model is None:
            model = self.config.model
            if model is None:
                raise ValueError("These results don't have a stored model")

            if etau.is_str(model):
                model = foz.load_zoo_model(model)

            self._model = model

        return self._model

    def compute_embeddings(
        self,
        samples,
        model=None,
        batch_size=None,
        num_workers=None,
        skip_failures=True,
        skip_existing=False,
        warn_existing=False,
        force_square=False,
        alpha=None,
    ):
        """Computes embeddings for the given samples using this backend's
        model.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
            model (None): a :class:`fiftyone.core.models.Model` to apply. If
                not provided, these results must have been created with a
                stored model, which will be used by default
            batch_size (None): an optional batch size to use when computing
                embeddings. Only applicable when a ``model`` is provided
            num_workers (None): the number of workers to use when loading
                images. Only applicable when a Torch-based model is being used
                to compute embeddings
            skip_failures (True): whether to gracefully continue without
                raising an error if embeddings cannot be generated for a sample
            skip_existing (False): whether to skip generating embeddings for
                sample/label IDs that are already in the index
            warn_existing (False): whether to log a warning if any IDs already
                exist in the index
            force_square (False): whether to minimally manipulate the patch
                bounding boxes into squares prior to extraction. Only
                applicable when a ``model`` and ``patches_field`` are specified
            alpha (None): an optional expansion/contraction to apply to the
                patches before extracting them, in ``[-1, inf)``. If provided,
                the length and width of the box are expanded (or contracted,
                when ``alpha < 0``) by ``(100 * alpha)%``. For example, set
                ``alpha = 1.1`` to expand the boxes by 10%, and set
                ``alpha = 0.9`` to contract the boxes by 10%. Only applicable
                when a ``model`` and ``patches_field`` are specified

        Returns:
            a tuple of:

            -   a ``num_embeddings x num_dims`` array of embeddings
            -   a ``num_embeddings`` array of sample IDs
            -   a ``num_embeddings`` array of label IDs, if applicable, or else
                ``None``
        """
        if model is None:
            model = self.get_model()

        if skip_existing:
            if self.config.patches_field is not None:
                index_ids = self.label_ids
            else:
                index_ids = self.sample_ids

            if index_ids is not None:
                samples = fbu.skip_ids(
                    samples,
                    index_ids,
                    patches_field=self.config.patches_field,
                    warn_existing=warn_existing,
                )
            else:
                logger.warning(
                    "This index does not support skipping existing IDs"
                )

        return fbu.get_embeddings(
            samples,
            model=model,
            patches_field=self.config.patches_field,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
        )

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        """Builds a :class:`SimilarityIndex` from a JSON representation of it.

        Args:
            d: a JSON dict
            samples: the :class:`fiftyone.core.collections.SampleCollection`
                for the run
            config: the :class:`SimilarityConfig` for the run
            brain_key: the brain key

        Returns:
            a :class:`SimilarityIndex`
        """
        raise NotImplementedError("subclass must implement _from_dict()")


class DuplicatesMixin(object):
    """Mixin for :class:`SimilarityIndex` instances that support duplicate
    detection operations.

    Similarity backends can expose this mixin simply by implementing
    :meth:`_radius_neighbors`.
    """

    def __init__(self):
        self._thresh = None
        self._unique_ids = None
        self._duplicate_ids = None
        self._neighbors_map = None

    @property
    def thresh(self):
        """The threshold used by the last call to :meth:`find_duplicates` or
        :meth:`find_unique`.
        """
        return self._thresh

    @property
    def unique_ids(self):
        """A list of unique IDs from the last call to :meth:`find_duplicates`
        or :meth:`find_unique`.
        """
        return self._unique_ids

    @property
    def duplicate_ids(self):
        """A list of duplicate IDs from the last call to
        :meth:`find_duplicates` or :meth:`find_unique`.
        """
        return self._duplicate_ids

    @property
    def neighbors_map(self):
        """A dictionary mapping IDs to lists of ``(dup_id, dist)`` tuples from
        the last call to :meth:`find_duplicates`.
        """
        return self._neighbors_map

    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        """Returns the neighbors within the given distance threshold for the
        given query.

        This method should only return results from the current :meth:`view`.

        Args:
            query (None): the query, which can be any of the following:

                -   an ID or list of IDs for which to return neighbors
                -   an embedding or ``num_queries x num_dim`` array of
                    embeddings for which to return neighbors
                -   ``None``, in which case the neighbors for all points in the
                    current :meth:`view are returned

            thresh (None): the distance threshold to use
            return_dists (False): whether to return query-neighbor distances

        Returns:
            the query result, in one of the following formats:

                -   an ``(ids, dists)`` tuple, when ``return_dists`` is True
                -   ``ids``, when ``return_dists`` is False

            In the above, ``ids`` contains the IDs of the nearest neighbors, in
            one of the following formats:

                -   a list of nearest neighbor IDs, when a single query ID or
                    vector is provided
                -   a list of lists of nearest neighbor IDs, when multiple
                    query IDs/vectors is provided
                -   a list of arrays of the **integer indexes** (not IDs) of
                    nearest neighbor points for every vector in the index, when
                    no query is provided

            and ``dists`` contains the corresponding query-neighbor distances
            for each result in ``ids``
        """
        raise NotImplementedError(
            "subclass must implement _radius_neighbors()"
        )

    def find_duplicates(self, thresh=None, fraction=None):
        """Queries the index to find near-duplicate examples based on the
        provided parameters.

        Calling this method populates the :meth:`unique_ids`,
        :meth:`duplicate_ids`, :attr:`neighbors_map`, and :attr:`thresh`
        properties of this object with the results of the query.

        Use :meth:`duplicates_view` and :meth:`visualize_duplicates` to analyze
        the results generated by this method.

        Args:
            thresh (None): a distance threshold to use to determine duplicates.
                If specified, the non-duplicate set will be the (approximately)
                largest set such that all pairwise distances between
                non-duplicate examples are greater than this threshold
            fraction (None): a desired fraction of images/patches to tag as
                duplicates, in ``[0, 1]``. In this case ``thresh`` is
                automatically tuned to achieve the desired fraction of
                duplicates
        """
        if self.config.patches_field is not None:
            logger.info("Computing duplicate patches...")
            ids = self.current_label_ids
        else:
            logger.info("Computing duplicate samples...")
            ids = self.current_sample_ids

        # Detect duplicates
        if fraction is not None:
            num_keep = int(round(min(max(0, 1.0 - fraction), 1) * len(ids)))
            unique_ids, thresh = self._remove_duplicates_count(
                num_keep, ids, init_thresh=thresh
            )
        else:
            unique_ids = self._remove_duplicates_thresh(thresh, ids)

        _unique_ids = set(unique_ids)
        duplicate_ids = [_id for _id in ids if _id not in _unique_ids]

        # Locate nearest non-duplicate for each duplicate
        if unique_ids and duplicate_ids:
            if self.config.patches_field is not None:
                unique_view = self._samples.select_labels(
                    ids=unique_ids, fields=self.config.patches_field
                )
            else:
                unique_view = self._samples.select(unique_ids)

            with self.use_view(unique_view):
                nearest_ids, dists = self._kneighbors(
                    query=duplicate_ids, k=1, return_dists=True
                )

            neighbors_map = defaultdict(list)
            for dup_id, _ids, _dists in zip(duplicate_ids, nearest_ids, dists):
                neighbors_map[_ids[0]].append((dup_id, _dists[0]))

            neighbors_map = {
                k: sorted(v, key=lambda t: t[1])
                for k, v in neighbors_map.items()
            }
        else:
            neighbors_map = {}

        logger.info("Duplicates computation complete")

        self._thresh = thresh
        self._unique_ids = unique_ids
        self._duplicate_ids = duplicate_ids
        self._neighbors_map = neighbors_map

    def find_unique(self, count):
        """Queries the index to select a subset of examples of the specified
        size that are maximally unique with respect to each other.

        Calling this method populates the :meth:`unique_ids`,
        :meth:`duplicate_ids`, and :attr:`thresh` properties of this object
        with the results of the query.

        Use :meth:`unique_view` and :meth:`visualize_unique` to analyze the
        results generated by this method.

        Args:
            count: the desired number of unique examples
        """
        if self.config.patches_field is not None:
            logger.info("Computing unique patches...")
            ids = self.current_label_ids
        else:
            logger.info("Computing unique samples...")
            ids = self.current_sample_ids

        unique_ids, thresh = self._remove_duplicates_count(count, ids)

        _unique_ids = set(unique_ids)
        duplicate_ids = [_id for _id in ids if _id not in _unique_ids]

        logger.info("Uniqueness computation complete")

        self._thresh = thresh
        self._unique_ids = unique_ids
        self._duplicate_ids = duplicate_ids
        self._neighbors_map = None

    def _remove_duplicates_count(self, num_keep, ids, init_thresh=None):
        if init_thresh is not None:
            thresh = init_thresh
        else:
            thresh = 1

        if num_keep <= 0:
            logger.info(
                "threshold: -, kept: %d, target: %d", num_keep, num_keep
            )
            return set(), None

        if num_keep >= len(ids):
            logger.info(
                "threshold: -, kept: %d, target: %d", num_keep, num_keep
            )
            return set(ids), None

        thresh_lims = [0, None]
        num_target = num_keep
        num_keep = -1

        while True:
            keep_ids = self._remove_duplicates_thresh(thresh, ids)
            num_keep_last = num_keep
            num_keep = len(keep_ids)

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

        return keep_ids, thresh

    def _remove_duplicates_thresh(self, thresh, ids):
        nearest_inds = self._radius_neighbors(thresh=thresh)

        n = len(ids)
        keep = set(range(n))
        for ind in range(n):
            if ind in keep:
                keep -= {i for i in nearest_inds[ind] if i > ind}

        return [ids[i] for i in keep]

    def plot_distances(self, bins=100, log=False, backend="plotly", **kwargs):
        """Plots a histogram of the distance between each example and its
        nearest neighbor.

        If `:meth:`find_duplicates` or :meth:`find_unique` has been executed,
        the threshold used is also indicated on the plot.

        Args:
            bins (100): the number of bins to use
            log (False): whether to use a log scale y-axis
            backend ("plotly"): the plotting backend to use. Supported values
                are ``("plotly", "matplotlib")``
            **kwargs: keyword arguments for the backend plotting method

        Returns:
            one of the following:

            -   a :class:`fiftyone.core.plots.plotly.PlotlyNotebookPlot`, if
                you are working in a notebook context and the plotly backend is
                used
            -   a plotly or matplotlib figure, otherwise
        """
        metric = self.config.metric
        thresh = self.thresh

        _, dists = self._kneighbors(k=1, return_dists=True)
        dists = np.array([d[0] for d in dists])

        if backend == "matplotlib":
            return _plot_distances_mpl(
                dists, metric, thresh, bins, log, **kwargs
            )

        return _plot_distances_plotly(
            dists, metric, thresh, bins, log, **kwargs
        )

    def duplicates_view(
        self,
        type_field=None,
        id_field=None,
        dist_field=None,
        sort_by="distance",
        reverse=False,
    ):
        """Returns a view that contains only the duplicate examples and their
        corresponding nearest non-duplicate examples generated by the last call
        to :meth:`find_duplicates`.

        If you are analyzing patches, the returned view will be a
        :class:`fiftyone.core.patches.PatchesView`.

        The examples are organized so that each non-duplicate is immediately
        followed by all duplicate(s) that are nearest to it.

        Args:
            type_field (None): the name of a string field in which to store
                ``"nearest"`` and ``"duplicate"`` labels. The field is created
                if necessary
            id_field (None): the name of a string field in which to store the
                ID of the nearest non-duplicate for each example in the view.
                The field is created if necessary
            dist_field (None): the name of a float field in which to store the
                distance of each example to its nearest non-duplicate example.
                The field is created if necessary
            sort_by ("distance"): specifies how to sort the groups of duplicate
                examples. The supported values are:

                -   ``"distance"``: sort the groups by the distance between the
                    non-duplicate and its (nearest, if multiple) duplicate
                -   ``"count"``: sort the groups by the number of duplicate
                    examples

            reverse (False): whether to sort in descending order

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        if self.neighbors_map is None:
            raise ValueError(
                "You must first call `find_duplicates()` to generate results"
            )

        samples = self.view
        patches_field = self.config.patches_field
        neighbors_map = self.neighbors_map

        if patches_field is not None and not isinstance(
            samples, fop.PatchesView
        ):
            samples = samples.to_patches(patches_field)

        if sort_by == "distance":
            key = lambda kv: min(e[1] for e in kv[1])
        elif sort_by == "count":
            key = lambda kv: len(kv[1])
        else:
            raise ValueError(
                "Invalid sort_by='%s'; supported values are %s"
                % (sort_by, ("distance", "count"))
            )

        existing_ids = set(samples.values("id"))
        neighbors = [
            (k, v) for k, v in neighbors_map.items() if k in existing_ids
        ]

        ids = []
        types = {}
        nearest_ids = {}
        dists = {}
        for _id, duplicates in sorted(neighbors, key=key, reverse=reverse):
            ids.append(_id)
            types[_id] = "nearest"
            nearest_ids[_id] = _id
            dists[_id] = 0.0

            for dup_id, dist in duplicates:
                ids.append(dup_id)
                types[dup_id] = "duplicate"
                nearest_ids[dup_id] = _id
                dists[dup_id] = dist

        if type_field is not None:
            samples.set_values(type_field, types, key_field="id")

        if id_field is not None:
            samples.set_values(id_field, nearest_ids, key_field="id")

        if dist_field is not None:
            samples.set_values(dist_field, dists, key_field="id")

        return samples.select(ids, ordered=True)

    def unique_view(self):
        """Returns a view that contains only the unique examples generated by
        the last call to :meth:`find_duplicates` or :meth:`find_unique`.

        If you are analyzing patches, the returned view will be a
        :class:`fiftyone.core.patches.PatchesView`.

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        if self.unique_ids is None:
            raise ValueError(
                "You must first call `find_unique()` or `find_duplicates()` "
                "to generate results"
            )

        samples = self.view
        patches_field = self.config.patches_field
        unique_ids = self.unique_ids

        if patches_field is not None and not isinstance(
            samples, fop.PatchesView
        ):
            samples = samples.to_patches(patches_field)

        return samples.select(unique_ids)

    def visualize_duplicates(self, visualization, backend="plotly", **kwargs):
        """Generates an interactive scatterplot of the results generated by the
        last call to :meth:`find_duplicates`.

        The ``visualization`` argument can be any visualization computed on the
        same dataset (or subset of it) as long as it contains every
        sample/object in the view whose results you are visualizing.

        The points are colored based on the following partition:

            -   "duplicate": duplicate example
            -   "nearest": nearest neighbor of a duplicate example
            -   "unique": the remaining unique examples

        Edges are also drawn between each duplicate and its nearest
        non-duplicate neighbor.

        You can attach plots generated by this method to an App session via its
        :attr:`fiftyone.core.session.Session.plots` attribute, which will
        automatically sync the session's view with the currently selected
        points in the plot.

        Args:
            visualization: a
                :class:`fiftyone.brain.visualization.VisualizationResults`
                instance to use to visualize the results
            backend ("plotly"): the plotting backend to use. Supported values
                are ``("plotly", "matplotlib")``
            **kwargs: keyword arguments for the backend plotting method:

                -   "plotly" backend: :meth:`fiftyone.core.plots.plotly.scatterplot`
                -   "matplotlib" backend: :meth:`fiftyone.core.plots.matplotlib.scatterplot`

        Returns:
            a :class:`fiftyone.core.plots.base.InteractivePlot`
        """
        if self.neighbors_map is None:
            raise ValueError(
                "You must first call `find_duplicates()` to generate results"
            )

        samples = self.view
        duplicate_ids = self.duplicate_ids
        neighbors_map = self.neighbors_map
        patches_field = self.config.patches_field

        dup_ids = set(duplicate_ids)
        nearest_ids = set(neighbors_map.keys())

        with visualization.use_view(samples, allow_missing=True):
            if patches_field is not None:
                ids = visualization.current_label_ids
            else:
                ids = visualization.current_sample_ids

            labels = []
            for _id in ids:
                if _id in dup_ids:
                    label = "duplicate"
                elif _id in nearest_ids:
                    label = "nearest"
                else:
                    label = "unique"

                labels.append(label)

            if backend == "plotly":
                kwargs["edges"] = _build_edges(ids, neighbors_map)
                kwargs["edges_title"] = "neighbors"
                kwargs["labels_title"] = "type"

            return visualization.visualize(
                labels=labels,
                classes=["unique", "nearest", "duplicate"],
                backend=backend,
                **kwargs,
            )

    def visualize_unique(self, visualization, backend="plotly", **kwargs):
        """Generates an interactive scatterplot of the results generated by the
        last call to :meth:`find_unique`.

        The ``visualization`` argument can be any visualization computed on the
        same dataset (or subset of it) as long as it contains every
        sample/object in the view whose results you are visualizing.

        The points are colored based on the following partition:

            -   "unique": the unique examples
            -   "other": the other examples

        You can attach plots generated by this method to an App session via its
        :attr:`fiftyone.core.session.Session.plots` attribute, which will
        automatically sync the session's view with the currently selected
        points in the plot.

        Args:
            visualization: a
                :class:`fiftyone.brain.visualization.VisualizationResults`
                instance to use to visualize the results
            backend ("plotly"): the plotting backend to use. Supported values
                are ``("plotly", "matplotlib")``
            **kwargs: keyword arguments for the backend plotting method:

                -   "plotly" backend: :meth:`fiftyone.core.plots.plotly.scatterplot`
                -   "matplotlib" backend: :meth:`fiftyone.core.plots.matplotlib.scatterplot`

        Returns:
            a :class:`fiftyone.core.plots.base.InteractivePlot`
        """
        if self.unique_ids is None:
            raise ValueError(
                "You must first call `find_unique()` to generate results"
            )

        samples = self.view
        unique_ids = self.unique_ids
        patches_field = self.config.patches_field

        unique_ids = set(unique_ids)

        with visualization.use_view(samples, allow_missing=True):
            if patches_field is not None:
                ids = visualization.current_label_ids
            else:
                ids = visualization.current_sample_ids

            labels = []
            for _id in ids:
                if _id in unique_ids:
                    label = "unique"
                else:
                    label = "other"

                labels.append(label)

            return visualization.visualize(
                labels=labels,
                classes=["other", "unique"],
                backend=backend,
                **kwargs,
            )


def _unique_no_sort(values):
    seen = set()
    return [v for v in values if v not in seen and not seen.add(v)]


def _build_edges(ids, neighbors_map):
    inds_map = {_id: idx for idx, _id in enumerate(ids)}

    edges = []
    for nearest_id, duplicates in neighbors_map.items():
        nearest_ind = inds_map[nearest_id]
        for dup_id, _ in duplicates:
            dup_ind = inds_map[dup_id]
            edges.append((dup_ind, nearest_ind))

    return np.array(edges)


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
