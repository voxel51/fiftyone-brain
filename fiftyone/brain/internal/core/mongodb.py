"""
MongoDB similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np

import eta.core.utils as etau

from fiftyone import ViewField as F
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)


logger = logging.getLogger(__name__)


class MongoDBSimilarityConfig(SimilarityConfig):
    """Configuration for a MongoDB similarity instance.

    Args:
        embeddings_field (None): the name of the embeddings field to use
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        index_name (None): the name of the MongoDB vector index to use
        **kwargs: keyword arguments for :class:`SimilarityConfig`
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name=None,
        **kwargs,
    ):
        if embeddings_field is None:
            raise ValueError(
                "You must provide the name of the field that contains the "
                "embeddings for this index by passing the `embeddings_field` "
                "parameter here"
            )

        if index_name is None:
            raise ValueError(
                "Programmatically creating vector search indexes is not yet "
                "supported by MongoDB Atlas. You must first create the index "
                "in Atlas and then provide the `index_name` parameter here"
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.index_name = index_name

    @property
    def method(self):
        return "mongodb"

    @property
    def max_k(self):
        return 10000  # MongoDB limit

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)


class MongoDBSimilarity(Similarity):
    """MongoDB similarity factory.

    Args:
        config: a :class:`MongoDBSimilarityConfig`
    """

    def ensure_requirements(self):
        # Could validate that user is connected to an Atlas cluster here
        # eg Atlas clusters generally have hostnames which end in "mongodb.net"
        # https://stackoverflow.com/q/73180110
        pass

    def ensure_usage_requirements(self):
        pass

    def initialize(self, samples, brain_key):
        return MongoDBSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class MongoDBSimilarityIndex(SimilarityIndex):
    """Class for interacting with MongoDB similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`MongoDBSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`MongoDBSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        sample_ids, label_ids = self._parse_data(samples, config)
        self._sample_ids = sample_ids
        self._label_ids = label_ids

        super().__init__(samples, config, brain_key, backend=backend)

    @property
    def sample_ids(self):
        return self._sample_ids

    @property
    def label_ids(self):
        return self._label_ids

    @property
    def total_index_size(self):
        return len(self._sample_ids)

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
        _sample_ids, _label_ids, ii, _ = fbu.add_ids(
            sample_ids,
            label_ids,
            self._sample_ids,
            self._label_ids,
            patches_field=self.config.patches_field,
            overwrite=overwrite,
            allow_existing=allow_existing,
            warn_existing=warn_existing,
        )

        if ii.size == 0:
            return

        _embeddings = embeddings[ii, :]

        fbu.add_embeddings(
            self._samples,
            _embeddings,
            _sample_ids,
            _label_ids,
            self.config.embeddings_field,
            patches_field=self.config.patches_field,
        )

        self._sample_ids = _sample_ids
        self._label_ids = _label_ids

        if reload:
            super().reload()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        _sample_ids, _label_ids, rm_inds = fbu.remove_ids(
            sample_ids,
            label_ids,
            self._sample_ids,
            self._label_ids,
            patches_field=self.config.patches_field,
            allow_missing=allow_missing,
            warn_missing=warn_missing,
        )

        if rm_inds.size == 0:
            return

        if self.config.patches_field is not None:
            rm_sample_ids = None
            rm_label_ids = self._label_ids[rm_inds]
        else:
            rm_sample_ids = self._sample_ids[rm_inds]
            rm_label_ids = None

        fbu.remove_embeddings(
            self._samples,
            self.config.embeddings_field,
            sample_ids=rm_sample_ids,
            label_ids=rm_label_ids,
            patches_field=self.config.patches_field,
        )

        self._sample_ids = _sample_ids
        self._label_ids = _label_ids

        if reload:
            super().reload()

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        _embeddings, _sample_ids, _label_ids = fbu.get_embeddings(
            self._samples,
            patches_field=self.config.patches_field,
            embeddings_field=self.config.embeddings_field,
        )

        if label_ids is not None:
            if self.config.patches_field is None:
                raise ValueError("This index does not support label IDs")

            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

            inds = _get_inds(
                label_ids,
                _label_ids,
                "label",
                allow_missing,
                warn_missing,
            )

            embeddings = _embeddings[inds, :]
            sample_ids = _sample_ids[inds]
            label_ids = np.asarray(label_ids)
        elif sample_ids is not None:
            if etau.is_str(sample_ids):
                sample_ids = [sample_ids]

            if self.config.patches_field is not None:
                sample_ids = set(sample_ids)
                bools = [_id in sample_ids for _id in _sample_ids]
                inds = np.nonzero(bools)[0]
            else:
                inds = _get_inds(
                    sample_ids,
                    _sample_ids,
                    "sample",
                    allow_missing,
                    warn_missing,
                )

            embeddings = _embeddings[inds, :]
            sample_ids = _sample_ids[inds]
            if self.config.patches_field is not None:
                label_ids = _label_ids[inds]
            else:
                label_ids = None
        else:
            embeddings = _embeddings
            sample_ids = _sample_ids
            label_ids = _label_ids

        return embeddings, sample_ids, label_ids

    def reload(self):
        sample_ids, label_ids = self._parse_data(self._samples, self.config)
        self._sample_ids = sample_ids
        self._label_ids = label_ids

        super().reload()

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        if query is None:
            raise ValueError("MongoDB does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "MongoDB does not support least similarity queries"
            )

        if aggregation not in (None, "mean"):
            raise ValueError(
                f"MongoDB does not support {aggregation} aggregation"
            )

        if k is None:
            k = min(self.index_size, self.config.max_k)

        # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage
        num_candidates = min(10 * k, self.config.max_k)

        query = self._parse_neighbors_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = [query]

        if self.view != self._samples:
            if self.config.patches_field is not None:
                index_ids = list(self.current_label_ids)
            else:
                index_ids = list(self.current_sample_ids)
        else:
            index_ids = None

        ids = []
        dists = []
        for q in query:
            search = {
                "index": self.config.index_name,
                "path": self.config.embeddings_field,
                "limit": k,
                "numCandidates": num_candidates,
                "queryVector": q,
            }

            if index_ids is not None:
                search["filter"] = {"_id": {"$in": index_ids}}

            project = {"_id": 1}
            if return_dists:
                project["score"] = {"$meta": "vectorSearchScore"}

            pipeline = [{"$vectorSearch": search}, {"$project": project}]
            matches = list(self._samples._aggregate(pipeline=pipeline))

            ids.append([str(m["_id"]) for m in matches])
            if return_dists:
                dists.append([m["score"] for m in matches])

        if single_query:
            ids = ids[0]
            if return_dists:
                dists = dists[0]

        if return_dists:
            return ids, dists

        return ids

    def _parse_neighbors_query(self, query):
        if etau.is_str(query):
            query_ids = [query]
            single_query = True
        else:
            query = np.asarray(query)

            # Query by vector(s)
            if np.issubdtype(query.dtype, np.number):
                return query

            query_ids = list(query)
            single_query = False

        # Query by ID(s)
        embeddings = (
            self._table.to_pandas().set_index("id").loc[query_ids]["vector"]
        )
        query = np.array([emb for emb in embeddings])

        if single_query:
            query = query[0, :]

        return query

    @staticmethod
    def _parse_data(samples, config):
        if config.patches_field is not None:
            samples = samples.filter_labels(
                config.patches_field, F(config.embeddings_field).exists()
            )
        else:
            samples = samples.exists(config.embeddings_field)

        return fbu.get_ids(samples, patches_field=config.patches_field)

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)


def _get_inds(ids, index_ids, ftype, allow_missing, warn_missing):
    if etau.is_str(ids):
        ids = [ids]

    ids_map = {_id: i for i, _id in enumerate(index_ids)}

    inds = []
    bad_ids = []

    for _id in ids:
        idx = ids_map.get(_id, None)
        if idx is not None:
            inds.append(idx)
        else:
            bad_ids.append(_id)

    num_missing = len(bad_ids)

    if num_missing > 0:
        if not allow_missing:
            raise ValueError(
                "Found %d %s IDs (eg '%s') that are not present in the index"
                % (num_missing, ftype, bad_ids[0])
            )

        if warn_missing:
            logger.warning(
                "Ignoring %d %s IDs that are not present in the index",
                num_missing,
                ftype,
            )

    return np.array(inds)
