"""
MongoDB similarity backend.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

from bson import ObjectId
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
        index_name (None): the name of the MongoDB vector index to use or
            create. If none is provided, a new index will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean")``
        **kwargs: keyword arguments for
            :class:`fiftyone.brain.similarity.SimilarityConfig`
    """

    def __init__(self, index_name=None, metric="cosine", **kwargs):
        if kwargs.get("embeddings_field") is None and index_name is None:
            raise ValueError(
                "You must provide either the name of a field to read/write "
                "embeddings for this index by passing the `embeddings` "
                "parameter, or you must provide the name of an existing "
                "vector search index via the `index_name` parameter"
            )

        # @todo support this. Will likely require copying embeddings to a new
        # collection as vector search indexes do not yet support array fields
        if kwargs.get("patches_field") is not None:
            raise ValueError(
                "The MongoDB backend does not yet support patch embeddings"
            )

        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        super().__init__(**kwargs)

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
        fou.ensure_package("pymongo>=4.7")

    def ensure_usage_requirements(self):
        fou.ensure_package("pymongo>=4.7")

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
        super().__init__(samples, config, brain_key, backend=backend)

        self._dataset = samples._dataset
        self._sample_ids = None
        self._label_ids = None
        self._index = None
        self._initialize()

    @property
    def is_external(self):
        return False

    @property
    def total_index_size(self):
        if self._sample_ids is not None:
            return len(self._sample_ids)

        if self._dataset.media_type == fom.GROUP:
            samples = self._dataset.select_group_slices(_allow_mixed=True)
        else:
            samples = self._dataset

        patches_field = self.config.patches_field
        embeddings_field = self.config.embeddings_field

        if patches_field is not None:
            _, embeddings_path = self._dataset._get_label_field_path(
                patches_field, embeddings_field
            )
            samples = samples.filter_labels(
                patches_field, F(embeddings_field).exists()
            )
            return samples.count(embeddings_path)

        if samples.has_field(embeddings_field):
            return samples.exists(embeddings_field).count()

        return 0

    def _initialize(self):
        coll = self._dataset._sample_collection

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
                    "to use vector search indexes"
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
            # Index already exists
            self._index = True
        elif self.total_index_size > 0:
            # Embeddings already exist but the index hasn't been declared yet
            dimension = self._get_dimension()
            self._create_index(dimension)
        else:
            # Index will be created when add_to_index() is called
            pass

    def _get_dimension(self):
        if self._dataset.media_type == fom.GROUP:
            samples = self._dataset.select_group_slices(_allow_mixed=True)
        else:
            samples = self._dataset

        patches_field = self.config.patches_field
        embeddings_field = self.config.embeddings_field

        if patches_field is not None:
            _, embeddings_path = self._dataset._get_label_field_path(
                patches_field, embeddings_field
            )
            view = samples.filter_labels(
                patches_field, F(embeddings_field).exists()
            ).limit(1)
            embeddings = view.values(embeddings_path, unwind=True)
        else:
            view = samples.exists(embeddings_field).limit(1)
            embeddings = view.values(embeddings_field)

        embedding = next(iter(embeddings), None)
        if embedding is None:
            return None

        return len(embedding)  # MongoDB requires list fields

    def _create_index(self, dimension):
        # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage
        # https://www.mongodb.com/docs/languages/python/pymongo-driver/current/indexes/atlas-search-index/
        from pymongo.operations import SearchIndexModel

        field = self._dataset.get_field(self.config.embeddings_field)
        if field is not None and not isinstance(field, fof.ListField):
            raise ValueError(
                "MongoDB vector search indexes require embeddings to be "
                "stored in list fields"
            )

        metric = _SUPPORTED_METRICS[self.config.metric]

        fields = [
            {
                "type": "vector",
                "numDimensions": dimension,
                "path": self.config.embeddings_field,
                "similarity": metric,
            },
            {
                "type": "filter",
                "path": "_id",
            },
        ]

        """
        if self._dataset.media_type == fom.GROUP:
            fields.append(
                {
                    "type": "filter",
                    "path": self._dataset.group_field + ".name",
                }
            )
        """

        model = SearchIndexModel(
            name=self.config.index_name,
            type="vectorSearch",  # requires pymongo>=4.7
            definition={"fields": fields},
        )

        coll = self._dataset._sample_collection
        coll.create_search_index(model=model)

        self._index = True

    @property
    def ready(self):
        """Returns True/False whether the vector search index is ready to be
        queried.
        """
        if self._index is None:
            return False

        try:
            coll = self._dataset._sample_collection
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

        if not overwrite or not allow_existing or warn_existing:
            if self._sample_ids is not None:
                _sample_ids, _label_ids = self._sample_ids, self._label_ids
            else:
                _sample_ids, _label_ids = self._parse_data(
                    self._dataset, self.config
                )

            index_sample_ids, index_label_ids, ii, _ = fbu.add_ids(
                sample_ids,
                label_ids,
                _sample_ids,
                _label_ids,
                patches_field=self.config.patches_field,
                overwrite=overwrite,
                allow_existing=allow_existing,
                warn_existing=warn_existing,
            )

            self._sample_ids = index_sample_ids
            self._label_ids = index_label_ids

            if ii.size == 0:
                return

            embeddings = embeddings[ii, :]
            sample_ids = sample_ids[ii]
            label_ids = label_ids[ii] if label_ids is not None else None
        else:
            index_sample_ids = None
            index_label_ids = None

        fbu.add_embeddings(
            self._dataset,
            embeddings.tolist(),  # MongoDB requires list fields
            sample_ids,
            label_ids,
            self.config.embeddings_field,
            patches_field=self.config.patches_field,
        )

        if reload:
            super().reload()

        self._sample_ids = index_sample_ids
        self._label_ids = index_label_ids

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        if not allow_missing or warn_missing:
            if self._sample_ids is not None:
                _sample_ids, _label_ids = self._sample_ids, self._label_ids
            else:
                _sample_ids, _label_ids = self._parse_data(
                    self._dataset, self.config
                )

            index_sample_ids, index_label_ids, rm_inds = fbu.remove_ids(
                sample_ids,
                label_ids,
                _sample_ids,
                _label_ids,
                patches_field=self.config.patches_field,
                allow_missing=allow_missing,
                warn_missing=warn_missing,
            )

            self._sample_ids = index_sample_ids
            self._label_ids = index_label_ids

            if rm_inds.size == 0:
                return

            if self.config.patches_field is not None:
                sample_ids = None
                label_ids = _label_ids[rm_inds]
            else:
                sample_ids = _sample_ids[rm_inds]
                label_ids = None
        else:
            index_sample_ids = None
            index_label_ids = None

        fbu.remove_embeddings(
            self._dataset,
            self.config.embeddings_field,
            sample_ids=sample_ids,
            label_ids=label_ids,
            patches_field=self.config.patches_field,
        )

        if reload:
            super().reload()

        self._sample_ids = index_sample_ids
        self._label_ids = index_label_ids

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        if self._dataset.media_type == fom.GROUP:
            samples = self._dataset.select_group_slices(_allow_mixed=True)
        else:
            samples = self._dataset

        if sample_ids is not None:
            samples = samples.select(sample_ids)
        elif label_ids is not None:
            if self.config.patches_field is None:
                raise ValueError("This index does not support label IDs")

            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

            samples = samples.select_labels(
                ids=label_ids, fields=self.config.patches_field
            )

        _embeddings, _sample_ids, _label_ids = fbu.get_embeddings(
            samples,
            patches_field=self.config.patches_field,
            embeddings_field=self.config.embeddings_field,
        )

        if label_ids is not None:
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
        self._sample_ids = None
        self._label_ids = None

        super().reload()

    def cleanup(self):
        if self._index is None:
            return

        try:
            coll = self._dataset._sample_collection
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
            index_ids = self.current_sample_ids
            # if self.config.patches_field is not None:
            #     index_ids = self.current_label_ids
        else:
            index_ids = None

        dataset = self._dataset

        sample_ids = []
        label_ids = None
        # if self.config.patches_field is not None:
        #     label_ids = []
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
                search["filter"] = {
                    "_id": {"$in": [ObjectId(_id) for _id in index_ids]}
                }

            """
            elif dataset.media_type == fom.GROUP:
                # $vectorSearch must be the first stage in all pipelines, so we
                # have to incorporate slice selection as a $filter
                name_field = dataset.group_field + ".name"
                group_slice = self.view.group_slice or dataset.group_slice
                search["filter"] = {name_field: {"$eq": group_slice}}
            """

            project = {"_id": 1}
            # if self.config.patches_field is not None:
            #     project["_sample_id"] = 1
            if return_dists:
                project["score"] = {"$meta": "vectorSearchScore"}

            # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage
            pipeline = [{"$vectorSearch": search}, {"$project": project}]

            try:
                matches = list(
                    dataset._aggregate(
                        pipeline=pipeline, manual_group_select=True
                    )
                )
            except OperationFailure as e:
                if index_ids is not None:
                    raise OperationFailure(
                        "This legacy search index does not yet support views. "
                        "Please follow the instructions at "
                        "https://github.com/voxel51/fiftyone-brain/pull/248 "
                        "to upgrade it"
                    ) from e
                else:
                    raise e

            sample_ids.append([str(m["_id"]) for m in matches])
            # if self.config.patches_field is not None:
            #     sample_ids.append([str(m["_sample_id"]) for m in matches])
            #     label_ids.append([str(m["_id"]) for m in matches])

            if return_dists:
                dists.append([m["score"] for m in matches])

        if single_query:
            sample_ids = sample_ids[0]
            if label_ids is not None:
                label_ids = label_ids[0]
            if return_dists:
                dists = dists[0]

        if return_dists:
            return sample_ids, label_ids, dists

        return sample_ids, label_ids

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
        if self._dataset.media_type == fom.GROUP:
            samples = self._dataset.select_group_slices(_allow_mixed=True)
        else:
            samples = self._dataset

        patches_field = self.config.patches_field
        embeddings_field = self.config.embeddings_field
        if patches_field is not None:
            _, embeddings_path = self._dataset._get_label_field_path(
                patches_field, embeddings_field
            )
            view = samples.filter_labels(
                patches_field, F("_id").is_in(query_ids)
            )
            embeddings = view.values(embeddings_path, unwind=True)
        else:
            view = samples.select(query_ids)
            embeddings = view.values(embeddings_field)

        return embeddings

    @staticmethod
    def _parse_data(samples, config):
        if samples.media_type == fom.GROUP:
            samples = samples.select_group_slices(_allow_mixed=True)

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
