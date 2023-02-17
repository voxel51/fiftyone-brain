"""
Sklearn similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

import eta.core.utils as etau

from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityResults,
)
import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)

_AGGREGATIONS = {"mean": np.mean, "min": np.min, "max": np.max}

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory
_COSINE_HACK_ATTR = "_cosine_hack"


class SklearnSimilarityConfig(SimilarityConfig):
    """Configuration for the sklearn similarity backend.

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

    @property
    def supports_least_similarity(self):
        return True

    @property
    def supports_aggregate_queries(self):
        return True

    @property
    def max_k(self):
        return None


class SklearnSimilarity(Similarity):
    """Sklearn similarity factory.

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
    """Class for interacting with sklearn similarity indexes.

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
        return self._sample_ids

    @property
    def label_ids(self):
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

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        if label_ids is not None:
            if self.config.patches_field is None:
                raise ValueError("This index does not support label IDs")

            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

            inds = _get_inds(
                label_ids,
                self.label_ids,
                "label",
                allow_missing,
                warn_missing,
            )

            embeddings = self._embeddings[inds, :]
            sample_ids = self.sample_ids[inds]
            label_ids = np.asarray(label_ids)
        elif sample_ids is not None:
            if etau.is_str(sample_ids):
                sample_ids = [sample_ids]

            if self.config.patches_field is not None:
                sample_ids = set(sample_ids)
                bools = [_id in sample_ids for _id in self.sample_ids]
                inds = np.nonzero(bools)[0]
            else:
                inds = _get_inds(
                    sample_ids,
                    self.sample_ids,
                    "sample",
                    allow_missing,
                    warn_missing,
                )

            embeddings = self._embeddings[inds, :]
            sample_ids = self.sample_ids[inds]
            if self.config.patches_field is not None:
                label_ids = self.label_ids[inds]
            else:
                label_ids = None
        else:
            embeddings = self._embeddings.copy()
            sample_ids = self.sample_ids.copy()
            if self.config.patches_field is not None:
                label_ids = None
            else:
                label_ids = self.label_ids.copy()

        return embeddings, sample_ids, label_ids

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

            if self.config.embeddings_field is not None:
                # @todo `_samples` is not not declared in SimilarityResults API
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
        print("Kneighbors")
        if aggregation is not None:
            print("Aggregation")
            print(aggregation)
            return self._sort_by_similarity(
                query, k, reverse, aggregation, return_dists
            )

        if keep_ids is not None:
            print("Keep IDs")
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
        print("\n================================")
        print(min_inds)

        if return_dists:
            return min_inds, min_dists
        return min_inds

    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        print("radius neighbors")
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
        # print(query)
        if query is None:
            raise ValueError(
                "A query must be provided when using aggregate similarity"
            )

        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                "Unsupported aggregation method '%s'. Supported values are %s"
                % (aggregation, tuple(_AGGREGATIONS.keys()))
            )
        print(self._curr_sample_ids)
        print(self._curr_label_ids)
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
                "Found %d %s IDs (eg %s) that are not present in the index"
                % (num_missing, ftype, bad_ids[0])
            )

        if warn_missing:
            logger.warning(
                "Ignoring %d %s IDs that are not present in the index",
                num_missing,
                ftype,
            )

    return np.array(inds)