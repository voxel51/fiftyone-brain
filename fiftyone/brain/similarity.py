"""
Similarity interface.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob


_INTERNAL_MODULE = "fiftyone.brain.internal.core.similarity"


class SimilarityResults(fob.BrainResults):
    """Class for performing similarity searches on a dataset.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` for
            which this index was computed
        embeddings: a ``num_embeddings x num_dims`` array of embeddings
        config: the :class:`SimilarityConfig` used to index the samples
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
        metric="euclidean",
        aggregation="mean",
    ):
        """Returns a view that sorts the samples/labels in the collection by
        visual similarity to the specified query.

        Args:
            query_ids: an ID or iterable of query IDs
            k (None): the number of matches to return. By default, the entire
                collection is sorted
            reverse (False): whether to sort by least similarity
            metric ("euclidean"): the distance metric to use. This parameter is
                passed directly to
                ``sklearn.metrics.pairwise_distances(..., metric=metric)``
            aggregation ("mean"): the aggregation method to use to compute
                composite similarities. Only applicable when ``query_ids``
                contains multiple IDs. Supported values are
                ``("mean", "min", "max")``

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        import fiftyone.brain.internal.core.similarity as fbs

        return fbs.sort_by_similarity(
            self._samples,
            self.embeddings,
            query_ids,
            self._sample_ids,
            label_ids=self._label_ids,
            patches_field=self.config.patches_field,
            k=k,
            reverse=reverse,
            metric=metric,
            aggregation=aggregation,
        )

    @classmethod
    def _from_dict(cls, d, samples):
        import fiftyone.brain.internal.core.similarity as fbs

        embeddings = np.array(d["embeddings"])
        config = fbs.SimilarityConfig.from_dict(d["config"])
        return cls(samples, embeddings, config)


class SimilarityConfig(fob.BrainMethodConfig):
    """Similarity configuration.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        patches_field (None): the sample field defining the patches we're
            indexing
    """

    def __init__(self, embeddings_field=None, patches_field=None, **kwargs):
        super().__init__(**kwargs)
        self.embeddings_field = embeddings_field
        self.patches_field = patches_field

    @property
    def method(self):
        return "similarity"

    @property
    def run_cls(self):
        run_cls_name = self.__class__.__name__[: -len("Config")]
        return etau.get_class(_INTERNAL_MODULE + "." + run_cls_name)


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
