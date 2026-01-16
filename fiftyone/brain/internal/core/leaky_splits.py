"""
Finds leaks between splits.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.validation as fov
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import fiftyone.brain as fb
import fiftyone.brain.similarity as fbs
import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "resnet18-imagenet-torch"
_DEFAULT_BATCH_SIZE = None


def compute_leaky_splits(
    samples,
    splits,
    threshold=None,
    roi_field=None,
    embeddings=None,
    similarity_index=None,
    model=None,
    model_kwargs=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    progress=None,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_data_field(
            samples,
            embeddings,
            data_type="embeddings",
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if etau.is_str(similarity_index):
        similarity_index = samples.load_brain_results(similarity_index)

    if (
        model is None
        and embeddings is None
        and similarity_index is None
        and not embeddings_exist
    ):
        model = foz.load_zoo_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = LeakySplitsConfig(
        splits=splits,
        embeddings_field=embeddings_field,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
    )

    brain_method = config.build()
    brain_method.ensure_requirements()

    if similarity_index is None:
        similarity_index = fb.compute_similarity(
            samples,
            backend="sklearn",
            roi_field=roi_field,
            embeddings=embeddings_field or embeddings,
            model=model,
            model_kwargs=model_kwargs,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=progress,
        )
    elif not isinstance(similarity_index, fbs.DuplicatesMixin):
        raise ValueError(
            "This method only supports similarity indexes that implement the "
            "%s mixin" % fbs.DuplicatesMixin
        )

    split_views = _to_split_views(samples, splits)

    index = brain_method.initialize(samples, similarity_index, split_views)

    if threshold is not None:
        index.find_leaks(threshold)

    return index


class LeakySplitsConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        splits=None,
        embeddings_field=None,
        similarity_index=None,
        model=None,
        model_kwargs=None,
        **kwargs,
    ):
        if isinstance(splits, dict):
            splits = None

        if similarity_index is not None and not etau.is_str(similarity_index):
            similarity_index = similarity_index.key

        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.splits = splits
        self.embeddings_field = embeddings_field
        self.similarity_index = similarity_index
        self.model = model
        self.model_kwargs = model_kwargs

        super().__init__(**kwargs)

    @property
    def type(self):
        return "leakage"

    @property
    def method(self):
        return "similarity"


class LeakySplits(fob.BrainMethod):
    def initialize(self, samples, similarity_index, split_views):
        return LeakySplitsIndex(
            samples, self.config, similarity_index, split_views
        )

    def get_fields(self, samples, _):
        fields = []
        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields


class LeakySplitsIndex(fob.BrainResults):
    def __init__(self, samples, config, similarity_index, split_views):
        super().__init__(samples, config, None)

        self._similarity_index = similarity_index
        self._split_views = split_views
        self._id2split = None
        self._thresh = None
        self._leak_ids = None

        self._initialize()

    @property
    def split_views(self):
        """A dict mapping split names to views."""
        return self._split_views

    @property
    def thresh(self):
        """The threshold used by the last call to :meth:`find_leaks`."""
        return self._thresh

    @property
    def leak_ids(self):
        """The list of leaky sample IDs from the last call to
        :meth:`find_leaks`.
        """
        return self._leak_ids

    def find_leaks(self, thresh):
        """Scans the index for leaks between splits.

        Args:
            thresh: the similarity distance threshold to use when detecting
                potential leaks
        """
        if thresh == self._thresh:
            return

        # Find duplicates
        self._thresh = thresh
        if self._similarity_index.thresh != self._thresh:
            self._similarity_index.find_duplicates(self._thresh)

        # Filter duplicates to just those with neighbors in different splits
        leak_ids = []
        neighbors_map = self._similarity_index.neighbors_map
        for sample_id, neighbors in neighbors_map.items():
            _leak_ids = []

            sample_split = self._id2split.get(sample_id, None)
            if sample_split is None:
                continue

            for n in neighbors:
                neighbor_id = n[0]
                neighbor_split = self._id2split.get(neighbor_id, None)
                if neighbor_split is None:
                    continue

                if neighbor_split != sample_split:
                    _leak_ids.append(neighbor_id)

            if _leak_ids:
                leak_ids.append(sample_id)
                leak_ids.extend(_leak_ids)

        self._leak_ids = leak_ids

    def leaks_view(self):
        """Returns a view containg all potential leaks generated by the last
        call to :meth:`find_leaks`.

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        if self._thresh is None:
            raise ValueError("You must first call `find_leaks()`")

        return self.samples.select(self._leak_ids, ordered=True)

    def leaks_for_sample(self, sample_or_id):
        """Returns a view that contains all leaks related to the given sample.

        The given sample is always first in the returned view, followed by any
        related leaks.

        Args:
            sample_or_id: a :class:`fiftyone.core.sample.Sample` or sample ID

        Returns:
            a :class:`fiftyone.core.view.DatasetView`
        """
        if self._thresh is None:
            raise ValueError("You must first call `find_leaks()`")

        if etau.is_str(sample_or_id):
            sample_id = sample_or_id
        else:
            sample_id = sample_or_id.id

        sample_split = self._id2split[sample_id]
        neighbors_map = self._similarity_index.neighbors_map

        leak_ids = []
        if sample_id in neighbors_map.keys():
            neighbors = neighbors_map[sample_id]
            leak_ids = [
                n[0] for n in neighbors if self._id2split[n[0]] != sample_split
            ]
        else:
            for unique_id, neighbors in neighbors_map.items():
                if sample_id in [n[0] for n in neighbors]:
                    leak_ids = [
                        n[0]
                        for n in neighbors
                        if self._id2split[n[0]] != sample_split
                    ]
                    leak_ids.append(unique_id)
                    break

        return self.samples.select([sample_id] + leak_ids, ordered=True)

    def no_leaks_view(self, view=None):
        """Returns a view with leaks excluded.

        Args:
            view (None): an optional :class:`fiftyone.core.view.DatasetView`
                from which to exclude. By default, :meth:`samples` is used
        """
        if self._thresh is None:
            raise ValueError("You must first call `find_leaks()`")

        if view is None:
            view = self.samples

        return view.exclude(self._leak_ids)

    def tag_leaks(self, tag="leak"):
        """Tags all potential leaks in :meth:`leaks_view` with the given tag.

        Args:
            tag ("leak"): the tag string to apply
        """
        self.leaks_view().tag_samples(tag)

    def _initialize(self):
        id2split = {}

        split_ids = {}
        for split_name, split_view in self.split_views.items():
            sample_ids = set(split_view.values("id"))
            split_ids[split_name] = sample_ids
            id2split.update({sid: split_name for sid in sample_ids})

        # Check for overlapping splits
        split_names = list(split_ids.keys())
        for idx, split1 in enumerate(split_names):
            for split2 in split_names[idx + 1 :]:
                overlap = split_ids[split1] & split_ids[split2]
                if overlap:
                    logger.warning(
                        "The '%s' and '%s' splits contain %d overlapping samples."
                        "Use dataset.match_tags('%s').match_tags('%s') to "
                        "identify them",
                        split1,
                        split2,
                        len(overlap),
                        split1,
                        split2,
                    )

        # Check for samples not in index
        index_ids = self._similarity_index.sample_ids
        if index_ids is not None:
            index_ids = set(index_ids)
            all_split_ids = set(id2split.keys())

            missing_ids = all_split_ids - index_ids
            if missing_ids:
                logger.warning(
                    "The provided splits contain %d samples (eg '%s') that "
                    "are not present in the index",
                    len(missing_ids),
                    next(iter(missing_ids)),
                )

        self._id2split = id2split


def _to_split_views(samples, splits):
    if etau.is_container(splits):
        return {tag: samples.match_tags(tag) for tag in splits}

    if isinstance(splits, str):
        field = samples.get_field(splits)
        if isinstance(field, fof.ListField):
            return {
                value: samples.exists(splits).match(F(splits).contains(value))
                for value in samples.distinct(splits)
            }
        else:
            return {
                value: samples.match(F(splits) == value)
                for value in samples.distinct(splits)
            }
