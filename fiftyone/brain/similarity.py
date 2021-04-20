"""
Similarity interface.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import fiftyone.core.brain as fob


class SimilarityResults(fob.BrainResults):
    """Class for performing similarity searches on a dataset.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` for
            which this index was computed
        embeddings: a ``num_embeddings x num_dims`` array of embeddings
        config: the :class:`SimilarityConfig` used to index the samples
    """

    def __init__(self, samples, embeddings, config):
        self._samples = samples
        self.embeddings = embeddings
        self.config = config
        self._ids = _get_ids_for_embeddings(
            embeddings, samples, patches_field=config.patches_field
        )

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
            self._ids,
            query_ids,
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


def _get_ids_for_embeddings(embeddings, samples, patches_field=None):
    if patches_field is not None:
        ids = samples._get_label_ids(fields=patches_field)
    else:
        ids = samples.values("id")

    if len(ids) != len(embeddings):
        ptype = "label" if patches_field is not None else "sample"
        raise ValueError(
            "Number of %s IDs (%d) does not match number of embeddings "
            "(%d). You may have missing data/labels that you need to omit "
            "from your view" % (ptype, len(ids), len(embeddings))
        )

    return np.array(ids)
