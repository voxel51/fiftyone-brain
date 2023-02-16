"""
Sklearn similarity.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityResults,
)
import fiftyone.brain.internal.core.utils as fbu
import fiftyone.core.utils as fou

pinecone = fou.lazy_import("pinecone")


logger = logging.getLogger(__name__)

_AGGREGATIONS = {"mean": np.mean, "min": np.min, "max": np.max}

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory
_COSINE_HACK_ATTR = "_cosine_hack"

class PineconeSimilarityConfig(SimilarityConfig):
    """Configuration for the pinecone similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (False): whether this run supports prompt queries
        metric ("euclidean"): the embedding distance metric to use. Supported 
            values are "euclidean", "cosine", and "dotproduct".
    """
    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        index_name="testname",
        dimension=None,
        pod_type="p1",
        pods=1,
        replicas=1,
        api_key=None,
        environment=None,
        upsert_pagination=100,
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )
        self.metric = metric
        self.index_name = index_name
        self.dimension = dimension
        self.pod_type = pod_type
        self.pods = pods
        self.replicas = replicas
        self.api_key = api_key
        self.environment = environment
        self.upsert_pagination = upsert_pagination

    @property
    def method(self):
        return "pinecone"

class PineconeSimilarity(Similarity):
    """Pinecone similarity class for similarity backends.

    Args:
        config: an :class:`PineconeSimilarityConfig`
    """

    def ensure_requirements(self):
        ## do I need "pinecone" here?
        pass

    def initialize(self, samples):
        return PineconeSimilarityResults(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass

class PineconeSimilarityResults(SimilarityResults):
    """Class for interacting with pinecone similarity results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
        backend (None): a :class:`PineconeSimilarity` instance
    """

    def __init__(
        self,
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
        backend=None,
    ):
        embeddings, sample_ids, label_ids = self._parse_data(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )

        dimension = self._parse_dimension(
            embeddings,
            config
        )

        self._embeddings = embeddings
        self._dimension = dimension
        self._sample_ids = sample_ids
        self._label_ids = label_ids
        self._index_name = config.index_name
        self._pod_type = config.pod_type
        self._pods = config.pods
        self._replicas = config.replicas
        self._metric = config.metric   
        self._upsert_pagination = config.upsert_pagination 
        self._api_key = config.api_key
        self._environment = config.environment
        
        print("Initializing pinecone index")
        pinecone.init(config.api_key, config.environment)
        if self._index_name not in pinecone.list_indexes():
            print("Creating pinecone index")
            pinecone.create_index(
                self._index_name, 
                dimension=self._dimension, 
                metric=self._metric, 
                pod_type=self._pod_type,
                pods=self._pods,
                replicas=self._replicas
            )

        self._neighbors_helper = None

        super().__init__(samples, config, backend=backend)

    @property
    def sample_ids(self):
        """The sample IDs of the full index."""
        return self._sample_ids

    @property
    def label_ids(self):
        """The label IDs of the full index, or ``None`` if not applicable."""
        return self._label_ids

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        
        pinecone.init(
            api_key=self._api_key, 
            environment=self._environment, 
        )
        index = pinecone.Index(self._index_name)

        if self._label_ids is not None:
            self._label_ids = [lid for lid in self._label_ids if lid not in label_ids]
            index.delete(ids = label_ids)
        elif self._sample_ids is not None:
            self._sample_ids = [sid for sid in self._sample_ids if sid not in sample_ids]
            index.delete(ids = sample_ids)

    def _sort_by_similarity(
        self, query, k, reverse, aggregation, return_dists
    ):
        if reverse == True:
            raise ValueError(
                "Pinecone backend does not support reverse sorting"
            )
            
        if k is None:
            raise ValueError(
                "k must be provided when using pinecone similarity"
            )

        if k > 10000:
            raise ValueError(
                "k cannot be greater than 10000 when using pinecone similarity"
            )

        if query is None:
            raise ValueError(
                "A query must be provided when using aggregate similarity"
            )

        if aggregation is not None:
            raise ValueError(
                "Pinecone backend does not support aggregation"
            )

        sample_ids = self.current_sample_ids
        label_ids = self.current_label_ids

        pinecone.init(
            api_key=self._api_key, 
            environment=self._environment, 
        )
        index = pinecone.Index(self._index_name)

        if isinstance(query, np.ndarray):
            # Query by vectors
            query_embedding = query.tolist()
        else:
            query_id = query
            query_embedding = index.fetch(
                [query_id]
                )['vectors'][query_id]['values']

        if label_ids is not None:
            response = index.query(
                vector=query_embedding,
                top_k=k,
                filter={
                    "id": {"$in": label_ids}
                }
            )
        else:
            response = index.query(
                vector=query_embedding,
                top_k=min(k, 10000),
                filter={
                    "id": {"$in": sample_ids}
                }
            )

        print(response)

        ids = ['63ed9a7ba4c597b4abcc6711' '63ed9a7ba4c597b4abcc6717']

        if return_dists:
            dists = []
            return ids, dists
        else:
            return ids

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
    ):

        embeddings_list = [arr.tolist() for arr in embeddings]
        if label_ids is not None:
            id_dicts = [{"id": lid, "sample_id": sid} for lid, sid in zip(label_ids, sample_ids)]
            index_vectors = list(zip(label_ids, embeddings_list, id_dicts))
        else:
            id_dicts = [{"id": sid, "sample_id": sid} for sid in sample_ids]
            index_vectors = list(zip(sample_ids, embeddings_list,id_dicts))
        
        num_vectors = embeddings.shape[0]
        num_steps = int(np.ceil(num_vectors / self._upsert_pagination))

        pinecone.init(
            api_key=self._api_key, 
            environment=self._environment, 
        )
        index = pinecone.Index(self._index_name)

        for i in range(num_steps):
            min_ind = self._upsert_pagination * i
            max_ind = min(self._upsert_pagination * (i+1), num_vectors)
            index.upsert(index_vectors[min_ind:max_ind])

    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is not None:
            attrs = [
                attr
                for attr in attrs
                if attr not in ("embeddings", "sample_ids", "label_ids")
            ]

        return attrs

    def _parse_dimension(self, embeddings, config):
        if config.dimension is not None:
            return int(config.dimension)
        elif embeddings is not None:
            return int(embeddings.shape[1])
        return 0

    # def _reload(self, hard=False):
    #     if hard:
    #         # @todo reload embeddings from gridFS too?
    #         # @todo `_samples` is not not declared in SimilarityResults API
    #         if self.config.embeddings_field is not None:
    #             embeddings, sample_ids, label_ids = self._parse_data(
    #                 self._samples,
    #                 self.config,
    #             )

    #             self._embeddings = embeddings
    #             self._sample_ids = sample_ids
    #             self._label_ids = label_ids
    #             self._neighbors_helper = None

    #     self.use_view(self._curr_view)

   
    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        pass

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        keep_ids=None,
        aggregation=None,
        return_dists=False,
    ):
        pass

    def _to_inds(self, ids):
        pass

    def _ensure_neighbors(self):
        pass

    def _get_neighbors(self, full=False):
        pass

    @staticmethod
    def _parse_data(
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
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

        return embeddings, sample_ids, label_ids    

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

        config_attrs = ["index_name",
                        "pod_type",
                        "pods",
                        "replicas",
                        "metric",
                        "upsert_pagination",
                        "api_key",
                        "environment",
                        ]
        
        for attr in config_attrs:
            if attr in d:
                value = d.get("index_name", None)
                if value is not None:
                    config[attr] = value

        return cls(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )
    


class SklearnSimilarityConfig(SimilarityConfig):
    """Configuration for the scikit-learn similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (False): whether this run supports prompt queries
        metric ("euclidean"): the embedding distance metric to use. See
            ``sklearn.metrics.pairwise_distance`` for supported values
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        **kwargs,
    ):
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )
        self.metric = metric

    @property
    def method(self):
        return "sklearn"


class SklearnSimilarity(Similarity):
    """Sklearn similarity class for similarity backends.

    Args:
        config: an :class:`SklearnSimilarityConfig`
    """

    def ensure_requirements(self):
        pass

    def initialize(self, samples):
        return SklearnSimilarityResults(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass


class SklearnSimilarityResults(SimilarityResults):
    """Class for interacting with sklearn similarity results.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
        backend (None): a :class:`SklearnSimilarity` instance
    """

    def __init__(
        self,
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
        backend=None,
    ):
        embeddings, sample_ids, label_ids = self._parse_data(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )

        self._embeddings = embeddings
        self._sample_ids = sample_ids
        self._label_ids = label_ids
        self._neighbors_helper = None

        super().__init__(samples, config, backend=backend)

    @property
    def sample_ids(self):
        """The sample IDs of the full index."""
        return self._sample_ids

    @property
    def label_ids(self):
        """The label IDs of the full index, or ``None`` if not applicable."""
        return self._label_ids

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
    ):
        # @todo handle `embeddings_field` case
        _sample_ids, _label_ids, ii, jj = fbu.add_ids(
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

        _embeddings = self._embeddings
        n, d = _embeddings.shape
        m = jj[-1] - n

        if m > 0:
            _embeddings = np.concatenate(
                (_embeddings, np.empty((m, d), dtype=_embeddings.dtype))
            )

        _embeddings[jj, :] = embeddings[ii, :]

        self._embeddings = _embeddings
        self._sample_ids = _sample_ids
        self._label_ids = _label_ids

        self._reload()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        # @todo handle `embeddings_field` case
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

        _embeddings = np.delete(self._embeddings, rm_inds)

        self._embeddings = _embeddings
        self._sample_ids = _sample_ids
        self._label_ids = _label_ids

        self._reload()

    def reload(self):
        self._reload(hard=True)

    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is not None:
            attrs = [
                attr
                for attr in attrs
                if attr not in ("embeddings", "sample_ids", "label_ids")
            ]

        return attrs

    def _reload(self, hard=False):
        if hard:
            # @todo reload embeddings from gridFS too?
            # @todo `_samples` is not not declared in SimilarityResults API
            if self.config.embeddings_field is not None:
                embeddings, sample_ids, label_ids = self._parse_data(
                    self._samples,
                    self.config,
                )

                self._embeddings = embeddings
                self._sample_ids = sample_ids
                self._label_ids = label_ids
                self._neighbors_helper = None

        self.use_view(self._curr_view)

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        keep_ids=None,
        aggregation=None,
        return_dists=False,
    ):
        if aggregation is not None:
            return self._sort_by_similarity(
                query, k, reverse, aggregation, return_dists
            )

        if keep_ids is not None:
            # @todo remove need for `keep_ids?
            query_inds = self._to_inds(query)
            keep_inds = self._to_inds(keep_ids)
            neighbors, dists = self._get_neighbors(full=True)

            if dists is not None and query_inds is not None:
                # Use pre-computed distances
                _dists = dists[keep_inds, :][:, query_inds]

                min_inds = np.argmin(_dists, axis=0)
                min_dists = _dists[min_inds, range(len(query_inds))]
            else:
                _index = self._embeddings[keep_inds, :]

                if query_inds is not None:
                    _query = self._embeddings[query_inds, :]
                else:
                    _query = query

                neighbors = skn.NearestNeighbors(metric=self.config.metric)
                neighbors.fit(_index)

                min_dists, min_inds = neighbors.kneighbors(
                    _query, n_neighbors=k
                )
                min_inds = min_inds.ravel()
                min_dists = min_dists.ravel()
        else:
            # @todo mismatched inds when view != full samples?
            query_inds = self._to_inds(query)
            neighbors, dists = self._get_neighbors()

            if dists is not None and not isinstance(query, np.ndarray):
                # Use pre-computed distances
                if query_inds is not None:
                    _dists = dists[:, query_inds]
                    _cols = range(len(query_inds))
                else:
                    _dists = dists
                    _cols = range(dists.shape[1])

                min_inds = np.argmin(_dists, axis=0)
                min_dists = _dists[min_inds, _cols]
            else:
                if query_inds is not None:
                    _query = self._embeddings[query_inds, :]
                else:
                    _query = query

                min_dists, min_inds = neighbors.kneighbors(
                    X=_query, n_neighbors=k
                )
                min_inds = min_inds.ravel()
                min_dists = min_dists.ravel()

        if return_dists:
            return min_inds, min_dists

        return min_inds

    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        neighbors, _ = self._get_neighbors()

        # When not using brute force, we approximate cosine distance by
        # computing Euclidean distance on unit-norm embeddings.
        # ED = sqrt(2 * CD), so we need to scale the threshold appropriately
        if getattr(neighbors, _COSINE_HACK_ATTR, False):
            thresh = np.sqrt(2.0 * thresh)

        query_inds = self._to_inds(query)
        if query_inds is not None:
            _query = self._embeddings[query_inds, :]
        else:
            _query = query

        dists, inds = neighbors.radius_neighbors(X=_query, radius=thresh)

        if return_dists:
            return inds, dists

        return inds

    def _to_inds(self, ids):
        if ids is None or isinstance(ids, np.ndarray):
            return None

        if self.config.patches_field is not None:
            ids = self.label_ids
        else:
            ids = self.sample_ids

        ids_map = {_id: i for i, _id in enumerate(ids)}
        return np.array([ids_map[_id] for _id in ids])

    def _sort_by_similarity(
        self, query, k, reverse, aggregation, return_dists
    ):
        if query is None:
            raise ValueError(
                "A query must be provided when using aggregate similarity"
            )

        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                "Unsupported aggregation method '%s'. Supported values are %s"
                % (aggregation, tuple(_AGGREGATIONS.keys()))
            )

        sample_ids = self.current_sample_ids
        label_ids = self.current_label_ids
        keep_inds = self._current_inds
        patches_field = self.config.patches_field
        metric = self.config.metric

        if isinstance(query, np.ndarray):
            # Query by vectors
            query_embeddings = query

            index_embeddings = self._embeddings
            if keep_inds is not None:
                index_embeddings = index_embeddings[keep_inds]

            dists = skm.pairwise_distances(
                index_embeddings, query_embeddings, metric=metric
            )
        else:
            # Query by IDs
            query_ids = query

            # Parse query IDs (always using full index)
            if patches_field is None:
                ids = self.sample_ids
            else:
                ids = self.label_ids

            bad_ids = []
            query_inds = []
            for query_id in query_ids:
                _inds = np.where(ids == query_id)[0]
                if _inds.size == 0:
                    bad_ids.append(query_id)
                else:
                    query_inds.append(_inds[0])

            if bad_ids:
                raise ValueError(
                    "Query IDs %s were not included in this index" % bad_ids
                )

            # Perform query
            self._ensure_neighbors()
            dists = self._neighbors_helper.get_distances()
            if dists is not None:
                # Use pre-computed distances
                if keep_inds is not None:
                    dists = dists[keep_inds, :]

                dists = dists[:, query_inds]
            else:
                # Compute distances from embeddings
                index_embeddings = self._embeddings
                if keep_inds is not None:
                    index_embeddings = index_embeddings[keep_inds]

                query_embeddings = self._embeddings[query_inds]
                dists = skm.pairwise_distances(
                    index_embeddings, query_embeddings, metric=metric
                )

        agg_fcn = _AGGREGATIONS[aggregation]
        dists = agg_fcn(dists, axis=1)

        inds = np.argsort(dists)
        if reverse:
            inds = np.flip(inds)

        if k is not None:
            inds = inds[:k]

        if patches_field is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if return_dists:
            return ids[inds], dists[inds]

        return ids[inds]

    def _ensure_neighbors(self):
        if self._neighbors_helper is None:
            self._neighbors_helper = NeighborsHelper(
                self._embeddings, self.config.metric
            )

    def _get_neighbors(self, full=False):
        self._ensure_neighbors()

        keep_inds = None if full else self._current_inds
        return self._neighbors_helper.get_neighbors(keep_inds=keep_inds)

    @staticmethod
    def _parse_data(
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
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

        return embeddings, sample_ids, label_ids

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


class NeighborsHelper(object):
    def __init__(self, embeddings, metric):
        self.embeddings = embeddings
        self.metric = metric

        self._initialized = False
        self._full_dists = None
        self._curr_keep_inds = None
        self._curr_dists = None
        self._curr_neighbors = None

    def get_distances(self, keep_inds=None):
        self._init()

        if self._same_keep_inds(keep_inds):
            return self._curr_dists

        if keep_inds is not None:
            dists, _ = self._build(keep_inds=keep_inds, build_neighbors=False)
        else:
            dists = self._full_dists

        self._curr_keep_inds = keep_inds
        self._curr_dists = dists
        self._curr_neighbors = None

        return dists

    def get_neighbors(self, keep_inds=None):
        self._init()

        if self._curr_neighbors is not None and self._same_keep_inds(
            keep_inds
        ):
            return self._curr_neighbors, self._curr_dists

        dists, neighbors = self._build(keep_inds=keep_inds)

        self._curr_keep_inds = keep_inds
        self._curr_dists = dists
        self._curr_neighbors = neighbors

        return neighbors, dists

    def _same_keep_inds(self, keep_inds):
        if keep_inds is None and self._curr_keep_inds is None:
            return True

        if (
            isinstance(keep_inds, np.ndarray)
            and isinstance(self._curr_keep_inds, np.ndarray)
            and keep_inds.size == self._curr_keep_inds.size
            and (keep_inds == self._curr_keep_inds).all()
        ):
            return True

        return False

    def _init(self):
        if self._initialized:
            return

        # Pre-compute all pairwise distances if number of embeddings is small
        if len(self.embeddings) <= _MAX_PRECOMPUTE_DISTS:
            dists, _ = self._build_precomputed(
                self.embeddings, build_neighbors=False
            )
        else:
            dists = None

        self._initialized = True
        self._full_dists = dists
        self._curr_keep_inds = None
        self._curr_dists = dists
        self._curr_neighbors = None

    def _build(self, keep_inds=None, build_neighbors=True):
        # Use full distance matrix if available
        if self._full_dists is not None:
            if keep_inds is not None:
                dists = self._full_dists[keep_inds, :][:, keep_inds]
            else:
                dists = self._full_dists

            if build_neighbors:
                neighbors = skn.NearestNeighbors(metric="precomputed")
                neighbors.fit(dists)
            else:
                neighbors = None

            return dists, neighbors

        # Must build index
        embeddings = self.embeddings

        if keep_inds is not None:
            embeddings = embeddings[keep_inds]

        if len(embeddings) <= _MAX_PRECOMPUTE_DISTS:
            dists, neighbors = self._build_precomputed(
                embeddings, build_neighbors=build_neighbors
            )
        else:
            dists = None
            neighbors = self._build_graph(embeddings)

        return dists, neighbors

    def _build_precomputed(self, embeddings, build_neighbors=True):
        logger.info("Generating index...")

        # Center embeddings
        embeddings = np.asarray(embeddings)
        embeddings -= embeddings.mean(axis=0, keepdims=True)

        dists = skm.pairwise_distances(embeddings, metric=self.metric)

        if build_neighbors:
            neighbors = skn.NearestNeighbors(metric="precomputed")
            neighbors.fit(dists)
        else:
            neighbors = None

        logger.info("Index complete")

        return dists, neighbors

    def _build_graph(self, embeddings):
        logger.info(
            "Generating neighbors graph for %d embeddings; this may take "
            "awhile...",
            len(embeddings),
        )

        # Center embeddings
        embeddings = np.asarray(embeddings)
        embeddings -= embeddings.mean(axis=0, keepdims=True)

        metric = self.metric

        if metric == "cosine":
            # Nearest neighbors does not directly support cosine distance, so
            # we approximate via euclidean distance on unit-norm embeddings
            cosine_hack = True
            embeddings = skp.normalize(embeddings, axis=1)
            metric = "euclidean"
        else:
            cosine_hack = False

        neighbors = skn.NearestNeighbors(metric=metric)
        neighbors.fit(embeddings)

        setattr(neighbors, _COSINE_HACK_ATTR, cosine_hack)

        logger.info("Index complete")

        return neighbors
