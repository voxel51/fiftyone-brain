"""
Similarity interface.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.utils as fou

fbs = fou.lazy_import("fiftyone.brain.internal.core.similarity")


class SimilarityResults(fob.BrainResults):
    """Class storing the results of :meth:`fiftyone.brain.compute_similarity`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        embeddings: a ``num_embeddings x num_dims`` array of embeddings
        config: the :class:`SimilarityConfig` used
    """

    def __init__(self, samples, embeddings, config):
        sample_ids, label_ids = _get_ids_for_embeddings(
            embeddings, samples, patches_field=config.patches_field
        )

        self._samples = samples
        self.embeddings = embeddings
        self.config = config
        self._sample_ids = sample_ids
        self._label_ids = label_ids

    def sort_by_similarity(
        self,
        query_ids,
        k=None,
        reverse=False,
        samples=None,
        metric="euclidean",
        aggregation="mean",
        mongo=False,
    ):
        """Returns a view that sorts the samples/labels in the collection by
        visual similarity to the specified query.

        Args:
            query_ids: an ID or iterable of query IDs
            k (None): the number of matches to return. By default, the entire
                collection is sorted
            reverse (False): whether to sort by least similarity
            samples (None): an optional
                :class:`fiftyone.core.collections.SampleCollection` defining
                the samples to include in the sort. If provided, the returned
                view will only include samples from this collection that were
                indexed. If not provided, all indexed samples are used
            metric ("euclidean"): the distance metric to use. This parameter is
                passed directly to
                ``sklearn.metrics.pairwise_distances(..., metric=metric)``
            aggregation ("mean"): the aggregation method to use to compute
                composite similarities. Only applicable when ``query_ids``
                contains multiple IDs. Supported values are
                ``("mean", "min", "max")``
            mongo (False): whether to return the aggregation pipeline defining
                the sort rather than constructing the actual
                :class:`fiftyone.core.view.DatasetView`

        Returns:
            a :class:`fiftyone.core.view.DatasetView`, or, if
            ``mongo == True``, a MongoDB aggregation pipeline (list of dicts)
        """
        if samples is not None and samples != self._samples:
            filter_ids = True
        else:
            filter_ids = False
            samples = self._samples

        return fbs.sort_by_similarity(
            samples,
            self.embeddings,
            query_ids,
            self._sample_ids,
            label_ids=self._label_ids,
            patches_field=self.config.patches_field,
            filter_ids=filter_ids,
            k=k,
            reverse=reverse,
            metric=metric,
            aggregation=aggregation,
            mongo=mongo,
        )

    @classmethod
    def _from_dict(cls, d, samples):
        embeddings = np.array(d["embeddings"])
        config = SimilarityConfig.from_dict(d["config"])
        return cls(samples, embeddings, config)


class SimilarityConfig(fob.BrainMethodConfig):
    """Similarity configuration.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
    """

    def __init__(
        self, embeddings_field=None, model=None, patches_field=None, **kwargs
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.embeddings_field = embeddings_field
        self.model = model
        self.patches_field = patches_field
        super().__init__(**kwargs)

    @property
    def method(self):
        return "similarity"

    @property
    def run_cls(self):
        run_cls_name = self.__class__.__name__[: -len("Config")]
        return getattr(fbs, run_cls_name)


def _get_ids_for_embeddings(embeddings, samples, patches_field=None):
    if patches_field is not None:
        sample_ids = []
        label_ids = []
        for l in samples._get_selected_labels(fields=patches_field):
            sample_ids.append(l["sample_id"])
            label_ids.append(l["label_id"])

        sample_ids = np.array(sample_ids)
        label_ids = np.array(label_ids)
    else:
        sample_ids = np.array(samples.values("id"))
        label_ids = None

    if len(sample_ids) != len(embeddings):
        ptype = "label" if patches_field is not None else "sample"
        raise ValueError(
            "Number of %s IDs (%d) does not match number of embeddings "
            "(%d). You may have missing data/labels that you need to omit "
            "from your view" % (ptype, len(sample_ids), len(embeddings))
        )

    return sample_ids, label_ids
