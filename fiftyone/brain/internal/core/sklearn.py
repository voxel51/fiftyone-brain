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
    DuplicatesMixin,
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)

_AGGREGATIONS = {
    "mean": np.mean,
    "post-mean": np.nanmean,
    "post-min": np.nanmin,
    "post-max": np.nanmax,
}

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory
_COSINE_HACK_ATTR = "_cosine_hack"


class SklearnSimilarityConfig(SimilarityConfig):
    """Configuration for the sklearn similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        metric ("cosine"): the embedding distance metric to use. See
            ``sklearn.metrics.pairwise_distance`` for supported values
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="cosine",
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
    def max_k(self):
        return None

    @property
    def supports_least_similarity(self):
        return True

    @property
    def supported_aggregations(self):
        return tuple(_AGGREGATIONS.keys())


class SklearnSimilarity(Similarity):
    """Sklearn similarity factory.

    Args:
        config: an :class:`SklearnSimilarityConfig`
    """

    def initialize(self, samples, brain_key):
        return SklearnSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class SklearnSimilarityIndex(SimilarityIndex, DuplicatesMixin):
    """Class for interacting with sklearn similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SklearnSimilarityConfig` used
        brain_key: the brain key
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
        brain_key,
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
        self._ids_to_inds = None
        self._curr_ids_to_inds = None
        self._neighbors_helper = None

        SimilarityIndex.__init__(
            self, samples, config, brain_key, backend=backend
        )
        DuplicatesMixin.__init__(self)

    @property
    def is_external(self):
        return self.config.embeddings_field is None

    @property
    def embeddings(self):
        return self._embeddings

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
        sample_ids = np.asarray(sample_ids)
        label_ids = np.asarray(label_ids) if label_ids is not None else None

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

        _embeddings = embeddings[ii, :]

        if self.config.embeddings_field is not None:
            fbu.add_embeddings(
                self._samples,
                _embeddings,
                sample_ids[ii],
                label_ids[ii] if label_ids is not None else None,
                self.config.embeddings_field,
                patches_field=self.config.patches_field,
            )

        _e = self._embeddings

        n = _e.shape[0]
        if n == 0:
            _e = np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype)

        d = _e.shape[1]
        m = jj[-1] - n + 1

        if m > 0:
            if _e.size > 0:
                _e = np.concatenate((_e, np.empty((m, d), dtype=_e.dtype)))
            else:
                _e = np.empty_like(_embeddings)

        _e[jj, :] = _embeddings

        self._embeddings = _e
        self._sample_ids = _sample_ids
        self._label_ids = _label_ids
        self._ids_to_inds = None
        self._curr_ids_to_inds = None
        self._neighbors_helper = None

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

        if self.config.embeddings_field is not None:
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

        _embeddings = np.delete(self._embeddings, rm_inds)

        self._embeddings = _embeddings
        self._sample_ids = _sample_ids
        self._label_ids = _label_ids
        self._ids_to_inds = None
        self._curr_ids_to_inds = None
        self._neighbors_helper = None

        if reload:
            super().reload()

    def use_view(self, *args, **kwargs):
        self._curr_ids_to_inds = None
        return super().use_view(*args, **kwargs)

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
                label_ids = self.label_ids.copy()
            else:
                label_ids = None

        return embeddings, sample_ids, label_ids

    def reload(self):
        if self.config.embeddings_field is not None:
            embeddings, sample_ids, label_ids = self._parse_data(
                self._samples, self.config
            )

            self._embeddings = embeddings
            self._sample_ids = sample_ids
            self._label_ids = label_ids
            self._ids_to_inds = None
            self._curr_ids_to_inds = None
            self._neighbors_helper = None

        super().reload()

    def cleanup(self):
        pass

    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is None:
            attrs.extend(["embeddings", "sample_ids", "label_ids"])

        return attrs

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        if aggregation is not None:
            return self._kneighbors_aggregate(
                query, k, reverse, aggregation, return_dists
            )

        (
            query,
            query_inds,
            full_index,
            single_query,
        ) = self._parse_neighbors_query(query)

        can_use_dists = full_index or query_inds is not None
        neighbors, dists = self._get_neighbors(can_use_dists=can_use_dists)

        if dists is not None:
            # Use pre-computed distances
            if query_inds is not None:
                _dists = dists[query_inds, :]
            else:
                _dists = dists

            # note: this must gracefully ignore nans
            inds = _nanargmin(_dists, k=k)

            if return_dists:
                dists = [d[i] for i, d in zip(inds, _dists)]
            else:
                dists = None
        else:
            if return_dists:
                dists, inds = neighbors.kneighbors(
                    X=query, n_neighbors=k, return_distance=True
                )
                inds = list(inds)
                dists = list(dists)
            else:
                inds = neighbors.kneighbors(
                    X=query, n_neighbors=k, return_distance=False
                )
                inds = list(inds)
                dists = None

        return self._format_output(
            inds, dists, full_index, single_query, return_dists
        )

    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        (
            query,
            query_inds,
            full_index,
            single_query,
        ) = self._parse_neighbors_query(query)

        can_use_dists = full_index or query_inds is not None
        neighbors, dists = self._get_neighbors(can_use_dists=can_use_dists)

        # When not using brute force, we approximate cosine distance by
        # computing Euclidean distance on unit-norm embeddings.
        # ED = sqrt(2 * CD), so we need to scale the threshold appropriately
        if getattr(neighbors, _COSINE_HACK_ATTR, False):
            thresh = np.sqrt(2.0 * thresh)

        if dists is not None:
            # Use pre-computed distances
            if query_inds is not None:
                _dists = dists[query_inds, :]
            else:
                _dists = dists

            # note: this must gracefully ignore nans
            inds = [np.nonzero(d <= thresh)[0] for d in _dists]

            if return_dists:
                dists = [d[i] for i, d in zip(inds, _dists)]
            else:
                dists = None
        else:
            if return_dists:
                dists, inds = neighbors.radius_neighbors(
                    X=query, radius=thresh, return_distance=True
                )
            else:
                dists = None
                inds = neighbors.radius_neighbors(
                    X=query, radius=thresh, return_distance=False
                )

        return self._format_output(
            inds, dists, full_index, single_query, return_dists
        )

    def _kneighbors_aggregate(
        self, query, k, reverse, aggregation, return_dists
    ):
        if query is None:
            raise ValueError("Full index queries do not support aggregation")

        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                "Unsupported aggregation method '%s'. Supported values are %s"
                % (aggregation, tuple(_AGGREGATIONS.keys()))
            )

        query, query_inds, _, _ = self._parse_neighbors_query(query)

        # Pre-aggregation
        if aggregation == "mean":
            if query.shape[0] > 1:
                query = query.mean(axis=0, keepdims=True)
                query_inds = None

            aggregation = None

        can_use_dists = query_inds is not None
        _, dists = self._get_neighbors(
            can_use_neighbors=False, can_use_dists=can_use_dists
        )

        if dists is not None:
            # Use pre-computed distances
            dists = dists[query_inds, :]
        else:
            keep_inds = self._current_inds
            index_embeddings = self._embeddings
            if keep_inds is not None:
                index_embeddings = index_embeddings[keep_inds]

            dists = skm.pairwise_distances(
                query, index_embeddings, metric=self.config.metric
            )

        # Post-aggregation
        if aggregation is not None:
            # note: this must gracefully ignore nans
            agg_fcn = _AGGREGATIONS[aggregation]
            dists = agg_fcn(dists, axis=0)
        else:
            dists = dists[0, :]

        if can_use_dists:
            dists[np.isnan(dists)] = 0.0

        inds = np.argsort(dists)
        if reverse:
            inds = np.flip(inds)

        if k is not None:
            inds = inds[:k]

        if self.config.patches_field is not None:
            ids = self.current_label_ids
        else:
            ids = self.current_sample_ids

        ids = list(ids[inds])

        if return_dists:
            dists = list(dists[inds])
            return ids, dists

        return ids

    def _parse_neighbors_query(self, query):
        # Full index
        if query is None:
            return None, None, True, False

        if etau.is_str(query):
            query_ids = [query]
            single_query = True
        else:
            query = np.asarray(query)

            # Query vector(s)
            if np.issubdtype(query.dtype, np.number):
                single_query = query.ndim == 1
                if single_query:
                    query = query[np.newaxis, :]

                return query, None, False, single_query

            query_ids = list(query)
            single_query = False

        # Retrieve indices into active `dists` matrix, if possible
        ids_to_inds = self._get_ids_to_inds(full=False)
        query_inds = []
        for _id in query_ids:
            _ind = ids_to_inds.get(_id, None)
            if _ind is not None:
                query_inds.append(_ind)
            else:
                # At least one query ID is not in the active index
                query_inds = None
                break

        # Retrieve embeddings
        ids_to_inds = self._get_ids_to_inds(full=True)
        inds = []
        bad_ids = []
        for _id in query_ids:
            _ind = ids_to_inds.get(_id, None)
            if _ind is not None:
                inds.append(_ind)
            else:
                bad_ids.append(_id)

        inds = np.array(inds)

        if bad_ids:
            raise ValueError(
                "Query IDs %s do not exist in this index" % bad_ids
            )

        query = self._embeddings[inds, :]

        if query_inds is not None:
            query_inds = np.array(query_inds)

        return query, query_inds, False, single_query

    def _get_ids_to_inds(self, full=False):
        if full:
            if self._ids_to_inds is None:
                if self.config.patches_field is not None:
                    ids = self.label_ids
                else:
                    ids = self.sample_ids

                self._ids_to_inds = {_id: i for i, _id in enumerate(ids)}

            return self._ids_to_inds

        if self._curr_ids_to_inds is None:
            if self.config.patches_field is not None:
                ids = self.current_label_ids
            else:
                ids = self.current_sample_ids

            self._curr_ids_to_inds = {_id: i for i, _id in enumerate(ids)}

        return self._curr_ids_to_inds

    def _get_neighbors(self, can_use_neighbors=True, can_use_dists=True):
        if self._neighbors_helper is None:
            self._neighbors_helper = NeighborsHelper(
                self._embeddings, self.config.metric
            )

        return self._neighbors_helper.get_neighbors(
            keep_inds=self._current_inds,
            can_use_neighbors=can_use_neighbors,
            can_use_dists=can_use_dists,
        )

    def _format_output(
        self, inds, dists, full_index, single_query, return_dists
    ):
        if full_index:
            return (inds, dists) if return_dists else inds

        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        ids = [[index_ids[i] for i in _inds] for _inds in inds]
        if return_dists:
            dists = [list(d) for d in dists]

        if single_query:
            ids = ids[0]
            if return_dists:
                dists = dists[0]

        return (ids, dists) if return_dists else ids

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
    def _from_dict(cls, d, samples, config, brain_key):
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
            brain_key,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )


class NeighborsHelper(object):

    _UNAVAILABLE = "UNAVAILABLE"

    def __init__(self, embeddings, metric):
        self.embeddings = embeddings
        self.metric = metric

        self._initialized = False
        self._full_dists = None

        self._curr_keep_inds = None
        self._curr_neighbors = None
        self._curr_dists = None

    def get_neighbors(
        self,
        keep_inds=None,
        can_use_neighbors=True,
        can_use_dists=True,
    ):
        iokay = self._same_keep_inds(keep_inds)
        nokay = not can_use_neighbors or self._curr_neighbors is not None
        dokay = not can_use_dists or self._curr_dists is not None

        if iokay and nokay and dokay:
            neighbors = self._curr_neighbors
            dists = self._curr_dists
        else:
            neighbors, dists = self._build(
                keep_inds=keep_inds,
                can_use_neighbors=can_use_neighbors,
                can_use_dists=can_use_dists,
            )

            if not iokay:
                self._curr_keep_inds = keep_inds

            if self._curr_neighbors is None or not iokay:
                self._curr_neighbors = neighbors

            if self._curr_dists is None or not iokay:
                self._curr_dists = dists

        if not can_use_neighbors or neighbors is self._UNAVAILABLE:
            neighbors = None

        if not can_use_dists or dists is self._UNAVAILABLE:
            dists = None

        return neighbors, dists

    def _same_keep_inds(self, keep_inds):
        # This handles either argument being None
        return np.array_equal(keep_inds, self._curr_keep_inds)

    def _build(
        self, keep_inds=None, can_use_neighbors=True, can_use_dists=True
    ):
        if can_use_dists:
            if (
                self._full_dists is None
                and len(self.embeddings) <= _MAX_PRECOMPUTE_DISTS
            ):
                self._full_dists = self._build_dists(self.embeddings)

            if self._full_dists is not None:
                if keep_inds is not None:
                    dists = self._full_dists[keep_inds, :][:, keep_inds]
                else:
                    dists = self._full_dists
            elif (
                keep_inds is not None
                and len(keep_inds) <= _MAX_PRECOMPUTE_DISTS
            ):
                dists = self._build_dists(self.embeddings[keep_inds])
            else:
                dists = self._UNAVAILABLE
        else:
            dists = None

        if can_use_neighbors:
            if not isinstance(dists, np.ndarray):
                embeddings = self.embeddings
                if keep_inds is not None:
                    embeddings = embeddings[keep_inds]

                neighbors = self._build_neighbors(embeddings)
            else:
                neighbors = self._UNAVAILABLE
        else:
            neighbors = None

        return neighbors, dists

    def _build_dists(self, embeddings):
        logger.info("Generating index for %d embeddings...", len(embeddings))

        # Center embeddings
        embeddings = np.asarray(embeddings)
        embeddings -= embeddings.mean(axis=0, keepdims=True)

        dists = skm.pairwise_distances(embeddings, metric=self.metric)
        np.fill_diagonal(dists, np.nan)

        logger.info("Index complete")

        return dists

    def _build_neighbors(self, embeddings):
        logger.info(
            "Generating neighbors graph for %d embeddings...",
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


def _nanargmin(array, k=1):
    if k == 1:
        inds = np.nanargmin(array, axis=1)
        inds = [np.array([i]) for i in inds]
    else:
        inds = np.argsort(array, axis=1)
        inds = list(inds[:, :k])

    return inds
