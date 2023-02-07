"""
Similarity interface.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.utils as fou

fbs = fou.lazy_import("fiftyone.brain.internal.core.similarity")
fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")


class SimilarityResults(fob.BrainResults):
    """Class storing the results of :meth:`fiftyone.brain.compute_similarity`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
    """

    def __init__(
        self, samples, config, embeddings=None, sample_ids=None, label_ids=None
    ):
        if embeddings is None:
            embeddings, sample_ids, label_ids = fbu.get_embeddings(
                samples._dataset,
                patches_field=config.patches_field,
                embeddings_field=config.embeddings_field,
            )
        elif sample_ids is None:
            sample_ids, label_ids = fbu.get_ids(
                samples,
                patches_field=config.patches_field,
                data=embeddings,
                data_type="embeddings",
            )

        self._samples = samples
        self._config = config
        self.embeddings = embeddings
        self.sample_ids = sample_ids
        self.label_ids = label_ids

        self._last_view = None
        self._curr_view = None
        self._curr_sample_ids = None
        self._curr_label_ids = None
        self._curr_keep_inds = None
        self._curr_good_inds = None
        self._neighbors_helper = None
        self._thresh = None
        self._unique_ids = None
        self._duplicate_ids = None
        self._neighbors_map = None
        self._model = None

        self.use_view(samples)

    def __enter__(self):
        self._last_view = self.view
        return self

    def __exit__(self, *args):
        self.use_view(self._last_view)
        self._last_view = None

    @property
    def config(self):
        """The :class:`SimilarityConfig` for the results."""
        return self._config

    @property
    def index_size(self):
        """The number of active data points in the index.

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
        return len(self.embeddings)

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
    def view(self):
        """The :class:`fiftyone.core.collections.SampleCollection` against
        which results are currently being generated.

        If :meth:`use_view` has been called, this view may be different than
        the collection on which the full index was generated.
        """
        return self._curr_view

    @property
    def thresh(self):
        """The threshold used by the last call to :meth:`find_duplicates` or
        :meth:`find_unique`.
        """
        return self._thresh

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

        self._curr_view = sample_collection
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
        return fbs.values(self, path_or_expr)

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
        return fbs.plot_distances(self, bins, log, backend, **kwargs)

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

            k (None): the number of matches to return. By default, all
                samples/labels are included
            reverse (False): whether to sort by least similarity
            aggregation ("mean"): the aggregation method to use to compute
                composite similarities. Only applicable when ``query`` contains
                multiple queries. Supported values are
                ``("mean", "min", "max")``
            dist_field (None): the name of a float field in which to store the
                distance of each example to the specified query. The field is
                created if necessary

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        return fbs.sort_by_similarity(
            self, query, k, reverse, aggregation, dist_field, _mongo
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
        return fbs.find_duplicates(self, thresh, fraction)

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
        return fbs.find_unique(self, count)

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

        return fbs.duplicates_view(
            self, type_field, id_field, dist_field, sort_by, reverse
        )

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

        return fbs.unique_view(self)

    def visualize_duplicates(
        self, visualization=None, backend="plotly", **kwargs
    ):
        """Generates an interactive scatterplot of the results generated by the
        last call to :meth:`find_duplicates`.

        If provided, the ``visualization`` argument can be any visualization
        computed on the same dataset (or subset of it) as long as it contains
        every sample/object in the view whose results you are visualizing. If
        no ``visualization`` argument is provided and the embeddings
        have more than 3 dimensions, a 2D representation of the embeddings is
        computed via :meth:`fiftyone.brain.compute_visualization`.

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
            visualization (None): a
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

        return fbs.visualize_duplicates(self, visualization, backend, **kwargs)

    def visualize_unique(self, visualization=None, backend="plotly", **kwargs):
        """Generates an interactive scatterplot of the results generated by the
        last call to :meth:`find_unique`.

        If provided, the ``visualization`` argument can be any visualization
        computed on the same dataset (or subset of it) as long as it contains
        every sample/object in the view whose results you are visualizing. If
        no ``visualization`` argument is provided and the embeddings
        have more than 3 dimensions, a 2D representation of the embeddings is
        computed via :meth:`fiftyone.brain.compute_visualization`.

        The points are colored based on the following partition:

            -   "unique": the unique examples
            -   "other": the other examples

        You can attach plots generated by this method to an App session via its
        :attr:`fiftyone.core.session.Session.plots` attribute, which will
        automatically sync the session's view with the currently selected
        points in the plot.

        Args:
            visualization (None): a
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

        return fbs.visualize_unique(self, visualization, backend, **kwargs)

    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is not None:
            attrs = [
                attr
                for attr in attrs
                if attr not in ("embeddings", "sample_ids", "label_ids")
            ]

        return attrs

    @classmethod
    def _from_dict(cls, d, samples, config):
        embeddings = d.get("embeddings", None)
        if embeddings is not None:
            embeddings = np.array(embeddings)

        sample_ids = d.get("sample_ids", None)
        if sample_ids is not None:
            sample_ids = np.array(sample_ids)

        label_ids = d.get("label_ids", None)
        if label_ids is not None:
            label_ids = np.array(label_ids)

        return cls(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )


class SimilarityConfig(fob.BrainMethodConfig):
    """Similarity configuration.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        metric (None): the embedding distance metric used
        supports_prompts (False): whether this run supports prompt queries
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        metric=None,
        supports_prompts=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        self.metric = metric
        self.supports_prompts = supports_prompts
        super().__init__(**kwargs)

    @property
    def method(self):
        return "similarity"

    @property
    def run_cls(self):
        run_cls_name = self.__class__.__name__[: -len("Config")]
        return getattr(fbs, run_cls_name)
