"""
MongoDB similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from pymongo.errors import OperationFailure

import eta.core.utils as etau

from fiftyone import ViewField as F
import fiftyone.core.fields as fof
import fiftyone.core.media as fom
import fiftyone.core.utils as fou
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "cosine": "cosine",
    "dotproduct": "dotProduct",
    "euclidean": "euclidean",
}


class MongoDBSimilarityConfig(SimilarityConfig):
    """Configuration for a MongoDB similarity instance.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        index_name (None): the name of the MongoDB vector index to use or
            create. If none is provided, a new index will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean")``
        **kwargs: keyword arguments for :class:`SimilarityConfig`
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name=None,
        metric="cosine",
        **kwargs,
    ):
        if embeddings_field is None and index_name is None:
            raise ValueError(
                "You must provide either the name of a field to read/write "
                "embeddings for this index by passing the `embeddings` "
                "parameter, or you must provide the name of an existing "
                "vector search index via the `index_name` parameter"
            )

        # @todo support this. Will likely require copying embeddings to a new
        # collection as vector search indexes do not yet support array fields
        if patches_field is not None:
            raise ValueError(
                "The MongoDB backend does not yet support patch embeddings"
            )

        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.index_name = index_name
        self.metric = metric

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
        #
        # https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.create_search_index
        #
        # Could also validate that user is connected to an Atlas cluster here
        # eg Atlas clusters generally have hostnames which end in "mongodb.net"
        # https://stackoverflow.com/q/73180110
        #
        fou.ensure_package("pymongo>=4.5")

    def ensure_usage_requirements(self):
        fou.ensure_package("pymongo>=4.5")

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

        self._index = None
        self._initialize()

    @property
    def is_external(self):
        return False

    @property
    def sample_ids(self):
        return self._sample_ids

    @property
    def label_ids(self):
        return self._label_ids

    @property
    def total_index_size(self):
        return len(self._sample_ids)

    def _initialize(self):
        coll = self._samples._dataset._sample_collection

        try:
            indexes = {
                i["name"]: i
                for i in coll.aggregate([{"$listSearchIndexes": {}}])
            }
        except OperationFailure:
            # https://www.mongodb.com/docs/manual/release-notes/7.0/#atlas-search-index-management
            # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview
            if self.config.index_name is None:
                raise ValueError(
                    "You must be running MongoDB Atlas 7.0 or later in order "
                    "to programmatically create vector search indexes. If "
                    "you are running MongoDB Atlas 6.0.11 then you can still "
                    "use this feature if you first manually create a vector "
                    "search index and then provide its name via the "
                    "`index_name` parameter"
                )

            # Must assume index exists because we can't use pymongo to check...
            self._index = True

            return

        if self.config.index_name is None:
            root = self.config.embeddings_field
            index_name = fbu.get_unique_name(root, list(indexes.keys()))

            self.config.index_name = index_name
            self.save_config()
        elif self.config.embeddings_field is None:
            info = indexes.get(self.config.index_name, None)
            if info is None:
                raise ValueError(
                    "Index '%s' does not exist" % self.config.index_name
                )

            self.config.embeddings_field = next(
                iter(info["latestDefinition"]["mappings"]["fields"].keys())
            )
            self.save_config()

        if self.config.index_name in indexes:
            self._index = True

    def _create_index(self, dimension):
        field = self._samples.get_field(self.config.embeddings_field)
        if field is not None and not isinstance(field, fof.ListField):
            raise ValueError(
                "MongoDB vector search indexes require embeddings to be "
                "stored in list fields"
            )

        metric = _SUPPORTED_METRICS[self.config.metric]

        # https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector
        # https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.create_search_index
        coll = self._samples._dataset._sample_collection
        coll.create_search_index(
            {
                "name": self.config.index_name,
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            self.config.embeddings_field: {
                                "type": "knnVector",
                                "dimensions": dimension,
                                "similarity": metric,
                            }
                        },
                    }
                },
            }
        )

        self._index = True

    @property
    def ready(self):
        """Returns True/False whether the vector search index is ready to be
        queried.
        """
        if self._index is None:
            return False

        try:
            coll = self._samples._dataset._sample_collection
            indexes = {
                i["name"]: i
                for i in coll.aggregate([{"$listSearchIndexes": {}}])
            }
        except OperationFailure:
            # requires MongoDB Atlas 7.0 or later
            return None

        info = indexes.get(self.config.index_name, {})
        return info.get("status", None) == "READY"

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
        if self._index is None:
            self._create_index(embeddings.shape[1])

        sample_ids = np.asarray(sample_ids)
        label_ids = np.asarray(label_ids) if label_ids is not None else None

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

        fbu.add_embeddings(
            self._samples,
            embeddings[ii, :].tolist(),  # MongoDB requires list fields
            sample_ids[ii],
            label_ids[ii] if label_ids is not None else None,
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

    def cleanup(self):
        if self._index is None:
            return

        try:
            coll = self._samples._dataset._sample_collection
            coll.drop_search_index(self.config.index_name)
        except OperationFailure:
            # requires MongoDB Atlas 7.0 or later
            pass

        self._index = None

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

        if self.has_view:
            if self.config.patches_field is not None:
                index_ids = list(self.current_label_ids)
            else:
                index_ids = list(self.current_sample_ids)
        else:
            index_ids = None

        dataset = self._samples._dataset

        ids = []
        dists = []
        for q in query:
            search = {
                "index": self.config.index_name,
                "path": self.config.embeddings_field,
                "limit": k,
                "numCandidates": num_candidates,
                "queryVector": q.tolist(),
            }

            if index_ids is not None:
                search["filter"] = {"_id": {"$in": index_ids}}
            elif dataset.media_type == fom.GROUP:
                # $vectorSearch must be the first stage in all pipelines, so we
                # have to incorporate slice selection as a $filter
                name_field = dataset.group_field + ".name"
                group_slice = self._samples.group_slice or dataset.group_slice
                search["filter"] = {name_field: {"$eq": group_slice}}

            project = {"_id": 1}
            if return_dists:
                project["score"] = {"$meta": "vectorSearchScore"}

            # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage
            pipeline = [{"$vectorSearch": search}, {"$project": project}]
            matches = list(
                dataset._aggregate(pipeline=pipeline, manual_group_select=True)
            )

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
        embeddings = self._get_embeddings(query_ids)
        num_missing = len(query_ids) - len(embeddings)
        for e in embeddings:
            num_missing += int(e is None)

        if num_missing > 0:
            if single_query:
                raise ValueError("The query ID does not exist in this index")
            else:
                raise ValueError(
                    f"{num_missing} query IDs do not exist in this index"
                )

        query = np.array(embeddings)
        if single_query:
            query = query[0, :]

        return query

    def _get_embeddings(self, query_ids):
        dataset = self._samples._dataset
        patches_field = self.config.patches_field
        embeddings_field = self.config.embeddings_field
        if patches_field is not None:
            _, embeddings_path = dataset._get_label_field_path(
                patches_field, embeddings_field
            )
            view = dataset.filter_labels(
                patches_field, F("_id").is_in(query_ids)
            )
            embeddings = view.values(embeddings_path, unwind=True)
        else:
            view = dataset.select(query_ids)
            embeddings = view.values(embeddings_field)

        return embeddings

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
