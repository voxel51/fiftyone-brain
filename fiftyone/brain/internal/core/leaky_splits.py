"""
Finds leaks between splits.
"""

from collections import defaultdict
from copy import copy

import fiftyone as fo
from fiftyone import ViewField as F

# pylint: disable=no-member
import cv2

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.brain.similarity as sim
import fiftyone.brain.internal.core.sklearn as skl_sim
import fiftyone.brain.internal.core.duplicates as dups
import fiftyone.brain.internal.core.utils as fbu
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")

_DEFAULT_MODEL = "clip-vit-base32-torch"
_DEFAULT_BATCH_SIZE = None


def compute_leaky_splits(
    samples,
    brain_key,
    split_views=None,
    split_field=None,
    split_tags=None,
    threshold=0.2,
    embeddings_field=None,
    model=None,
    model_kwargs=None,
    patches_field=None,
    metric="cosine",
    **kwargs,
):
    """Uses embeddings to index the samples or their patches so that you can
    find leaks.

    Calling this method only creates the index. You can then call the methods
    exposed on the retuned object to perform the following operations:

    -   :meth:`leaks <fiftyone.brain.core.internal.leaky_splits.LeakySplitIndexInterface.leaks>`:
        Returns a view of all leaks in the dataset.

    -   :meth:`view_without_leaks <fiftyone.brain.core.internal.leaky_splits.LeakySplitIndexInterface.view_without_leaks>`:
        Returns a subset of the given view without any leaks.

    -   :meth:`tag_leaks <fiftyone.brain.core.internal.leaky_splits.LeakySplitIndexInterface.tag_leaks>`:
        Tags leaks in the dataset as leaks.


    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        split_views (None): a list of :class:`fiftyone.core.view.DatasetView`
            corresponding to different splits in the datset. Only one of
            `split_views`, `split_field`, and `splits_tags` need to be used.
        split_field (None): a string name of a field that holds the split of the sample.
            Each unique value in the field will be treated as a split.
            Only one of `split_views`, `split_field`, and `splits_tags` need to be used.
        split_tags (None): a list of strings, tags corresponding to differents splits.
            Only one of `split_views`, `split_field`, and `splits_tags` need to be used.
        threshold (0.2): The threshold to run the algorithm with. Values between
            0.1 - 0.25 tend to give good results.
        patches_field (None): a sample field defining the image patches in each
            sample that have been/will be embedded. Must be of type
            :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines`
        embeddings_field (None): field for embeddings to feed the index. This argument's
            behavior depends on whether a ``model`` is provided, as described
            below.

            If no ``model`` is provided, this argument specifies the field of pre-computed
            embeddings to use:

            If a ``model`` is provided, this argument specifies where to store
            the model's embeddings:
        brain_key (None): a brain key under which to store the results of this
            method
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/index.html>`_
            to use, or that was already used, to generate embeddings. The model
            must expose embeddings (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        **kwargs: keyword arguments for the
            :class:`fiftyone.brain.SklearnSimilarityIndex` class

    Returns:
        a :class:`fiftyone.brain.internal.core.leaky_splits.LeakySplitsSKLIndex`
    """

    fov.validate_collection(samples)

    embeddings_exist = False
    if embeddings_field is not None and model is None:
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings_field,
            patches_field=patches_field,
        )

    config = LeakySplitsSKLConfig(
        split_views=split_views,
        split_field=split_field,
        split_tags=split_tags,
        embeddings_field=embeddings_field,
        model=model,
        model_kwargs=model_kwargs,
        patches_field=patches_field,
        metric=metric,
        **kwargs,
    )
    brain_method = LeakySplitsSKL(config)
    brain_method.ensure_requirements()

    if brain_key is not None:
        # Don't allow overwriting an existing run with same key, since we
        # need the existing run in order to perform workflows like
        # automatically cleaning up the backend's index
        brain_method.register_run(samples, brain_key, overwrite=False)

    results = brain_method.initialize(samples, brain_key)

    results.set_threshold(threshold)
    leaks = results.leaks

    brain_method.save_run_results(samples, brain_key, results)

    return results, leaks


### GENERAL


class LeakySplitsConfigInterface(object):
    """Configuration for Leaky Splits

    Args:
        split_views (None): list of views corresponding to different splits
        split_field (None): field name that contains the split that the sample belongs to
        split_tags (None): list of tags that correspond to different splits
    """

    def __init__(
        self, split_views=None, split_field=None, split_tags=None, **kwargs
    ):
        self.split_views = split_views
        self.split_field = split_field
        self.split_tags = split_tags
        super().__init__(**kwargs)


class LeakySplitIndexInterface(object):
    """Interface for the index. To expose it, implement the property `leaks`.
    It shoud return a view of all leaks in the dataset.
    """

    def __init__(self) -> None:
        pass

    @property
    def num_leaks(self):
        """Returns the number of leaks found."""
        return self.leaks.count

    @property
    def leaks(self):
        """Returns view with all potential leaks."""
        raise NotImplementedError("Subclass must implement method.")

    def leaks_by_sample(self, sample):
        """Return view with all leaks related to a certain sample."""
        raise NotImplementedError("Subclass must implement method.")

    def view_without_leaks(self, view):
        return view.exclude([s["id"] for s in self.leaks])

    def tag_leaks(self, tag="leak"):
        """Tag leaks"""
        for s in self.leaks.iter_samples():
            s.tags.append(tag)
            s.save()

    def _id2split(self, sample_id, split_views):

        for i, split_view in enumerate(split_views):
            if len(split_view.select([sample_id])) > 0:
                return i

        return -1


def _to_views(samples, split_views=None, split_field=None, split_tags=None):
    """Helper function so that we can always work with views"""

    arithmetic_true = lambda x: int(x is not None)
    num_given = (
        arithmetic_true(split_views)
        + arithmetic_true(split_field)
        + arithmetic_true(split_tags)
    )

    if num_given == 0:
        raise ValueError(f"One of the split arguments must be given.")
    if num_given > 1:
        raise ValueError(f"Only one of the split arguments must be given.")

    if split_views:
        return split_views

    if split_field:
        return _field_to_views(samples, split_field)

    if split_tags:
        return _tags_to_views(samples, split_tags)


def _field_to_views(samples, field):
    field_values = samples.distinct(field)

    if len(field_values) < 2:
        raise ValueError(
            f"Field {field} has less than 2 distinct values,"
            f"can't be used to create splits"
        )

    views = []
    for val in field_values:
        view = samples.match(F(field) == val)
        views.append(view)

    return views


def _tags_to_views(samples, tags):
    if len(tags) < 2:
        raise ValueError("Must provide at least two tags.")

    views = []
    for tag in tags:
        view = samples.match_tags([tag])
        views.append(view)
    return views


###

### SKL BACKEND
class LeakySplitsSKLConfig(
    skl_sim.SklearnSimilarityConfig, LeakySplitsConfigInterface
):
    """Configuration for Leaky Splits with the SKLearn backend

    Args:
        split_views (None): list of views corresponding to different splits
        split_field (None): field name that contains the split that the sample belongs to
        split_tags (None): list of tags that correspond to different splits

        For the rest of the arguments, see :class:`SklearnSimilarityConfig`
    """

    def __init__(
        self,
        split_views=None,
        split_field=None,
        split_tags=None,
        embeddings_field=None,
        model=None,
        model_kwargs=None,
        patches_field=None,
        metric="cosine",
        **kwargs,
    ):
        LeakySplitsConfigInterface.__init__(
            self, split_views, split_field, split_tags
        )
        skl_sim.SklearnSimilarityConfig.__init__(
            self,
            embeddings_field=embeddings_field,
            model=model,
            model_kwargs=model_kwargs,
            patches_field=patches_field,
            metric=metric,
            **kwargs,
        )

    @property
    def method(self):
        return "Neural"


class LeakySplitsSKL(skl_sim.SklearnSimilarity):
    def initialize(self, samples, brain_key):
        return LeakySplitsSKLIndex(
            samples, self.config, brain_key, backend=self
        )


class LeakySplitsSKLIndex(
    skl_sim.SklearnSimilarityIndex, LeakySplitIndexInterface
):
    def __init__(self, samples, config, brain_key, **kwargs):
        skl_sim.SklearnSimilarityIndex.__init__(
            self, samples=samples, config=config, brain_key=brain_key, **kwargs
        )
        self.split_views = _to_views(
            samples,
            self.config.split_views,
            self.config.split_field,
            self.config.split_tags,
        )
        self._leak_threshold = 1

    def set_threshold(self, threshold):
        """Set threshold for leak computation"""
        self._leak_threshold = threshold

    @property
    def leaks(self):
        """
        Returns view with all potential leaks.
        """

        if not self.total_index_size == len(self._dataset):
            embeddings, sample_ids, label_ids = self.compute_embeddings(
                self._dataset
            )
            self.add_to_index(embeddings, sample_ids, label_ids)
        self.find_duplicates(self._leak_threshold)
        duplicates = self.duplicates_view()

        to_remove = []
        for sample_id, neighbors in self.neighbors_map.items():
            remove_sample = True
            sample_split = self._id2split(sample_id, self.split_views)
            for n in neighbors:
                if not (
                    self._id2split(n[0], self.split_views) == sample_split
                ):
                    remove_sample = False
                    if sample_split == 0:
                        neighbor_sample = self._dataset.select([n[0]]).first()
                        neighbor_sample.tags.append(f"leak with {sample_id}")
                        neighbor_sample.save()

            if remove_sample:
                to_remove.append(sample_id)

        duplicates = duplicates.exclude(to_remove)

        return duplicates


###

### HASH BACKEND

_HASH_METHODS = ["filepath", "image"]


class LeakySplitsHashConfig(fob.BrainMethodConfig, LeakySplitsConfigInterface):
    """
    Args:
        hash_field (None): string, field to write hashes into
    """

    def __init__(
        self,
        split_views=None,
        split_field=None,
        split_tags=None,
        method="filepath",
        hash_field=None,
        **kwargs,
    ):
        self._method = method
        self.hash_field = hash_field
        LeakySplitsConfigInterface.__init__(
            self,
            split_views=split_views,
            split_field=split_field,
            split_tags=split_tags,
        )
        fob.BrainMethodConfig.__init__(self, **kwargs)

    @property
    def method(self):
        return self._method


class LeakySplitsHash(fob.BrainMethod):
    def initialize(self, samples, brain_key):
        return LeakySplitsHashIndex(
            samples, self.config, brain_key, backend=self
        )


class LeakySplitsHashIndex(fob.BrainResults, LeakySplitIndexInterface):
    """ """

    def __init__(self, samples, config, brain_key, backend):
        fob.BrainResults.__init__(
            self, samples, config, brain_key, backend=backend
        )
        LeakySplitIndexInterface.__init__(self)
        self._hash2ids = defaultdict(list)
        self.split_views = _to_views(
            samples,
            self.config.split_views,
            self.config.split_field,
            self.config.split_tags,
        )
        self._dataset = samples._dataset
        self._compute_hashes(samples)

    @property
    def _hash_function(self):
        if self.config.method == "filepath":
            return fou.compute_filehash

        elif self.config.method == "image":
            return LeakySplitsHashIndex._image_hash

    def _compute_hashes(self, samples):
        for s in samples.iter_samples():
            hash = str(self._hash_function(s["filepath"]))
            self._hash2ids[hash].append(s["id"])
            if self.config.hash_field:
                s[self.config.hash_field] = hash
                s.save()

    @staticmethod
    def _image_hash(image, hash_size=24):
        """
        Compute the dHash for the input image.

        :param image: image filepath
        :param hash_size: Size of the hash (default 8x8).
        :return: The dHash value of the image as a 64-bit integer.
        """

        with open(image, "r"):
            image = cv2.imread(image)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to (hash_size + 1, hash_size)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        # Compute the differences between adjacent pixels
        diff = resized[:, 1:] > resized[:, :-1]

        # Convert the difference image to a binary hash
        # hash_value = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

        # Convert the difference image to a binary hash
        binary_string = "".join(["1" if v else "0" for v in diff.flatten()])

        # Convert the binary string to a hexadecimal string
        hex_hash = f"{int(binary_string, 2):0{hash_size * hash_size // 4}x}"

        return hex_hash

    @property
    def leaks(self):
        leak_ids = []
        for id_list in self._hash2ids.values():
            if len(id_list) > 1:
                leak_ids = leak_ids + id_list

        return self._dataset.select(leak_ids, ordered=True)

    def leaks_by_sample(self, sample):
        id = None
        if isinstance(sample, str):
            id = sample
        else:
            id = sample["id"]
        for id_list in self._hash2ids.values():
            if id in id_list:
                return self._dataset.select(id_list)


###
