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

import fiftyone as fo
import fiftyone.core.brain as fob
import fiftyone.brain.similarity as sim
import fiftyone.brain.internal.core.sklearn as skl_sim
import fiftyone.brain.internal.core.duplicates as dups
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")

_DEFAULT_MODEL = "clip-vit-base32-torch"
_DEFAULT_BATCH_SIZE = None


def compute_leaky_splits(
    samples,
    brain_key=None,
    split_views=None,
    split_field=None,
    split_tags=None,
    threshold=0.2,
    similarity_brain_key=None,
    embeddings_field=None,
    model=None,
    model_kwargs=None,
    similarity_backend=None,
    similarity_config_dict=None,
    **kwargs,
):
    """Uses embeddings to index the samples so that you can
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
        a :class:`fiftyone.brain.internal.core.leaky_splits.LeakySplitsSKLIndex`, a :class:`fiftyone.core.view.DatasetView`
    """

    fov.validate_collection(samples)

    config = LeakySplitsConfig(
        split_views,
        split_field,
        split_tags,
        similarity_brain_key,
        embeddings_field,
        model,
        model_kwargs,
        similarity_backend,
        similarity_config_dict,
        **kwargs,
    )
    brain_method = config.build()
    brain_method.ensure_requirements()

    if brain_key is not None:
        # Don't allow overwriting an existing run with same key, since we
        # need the existing run in order to perform workflows like
        # automatically cleaning up the backend's index
        brain_method.register_run(samples, brain_key, overwrite=False)

    results = brain_method.initialize(samples, brain_key)
    results.set_threshold(threshold)
    leaks = results.leaks

    if brain_key is not None:
        brain_method.save_run_results(samples, brain_key, results)
    if results._save_similarity_index:
        results.similarity_method.save_run_results(
            samples, similarity_brain_key, results.similarity_index
        )

    return results, leaks


### GENERAL


class LeakySplitsConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        split_views=None,
        split_field=None,
        split_tags=None,
        similarity_brain_key=None,
        embeddings_field=None,
        model=None,
        model_kwargs=None,
        similarity_backend=None,
        similarity_config_dict=None,
        **kwargs,
    ):
        self.split_views = split_views
        self.split_field = split_field
        self.split_tags = split_tags
        self.similarity_brain_key = similarity_brain_key
        self.embeddings_field = embeddings_field
        self.model = model
        self.model_kwargs = model_kwargs
        self.similarity_backend = similarity_backend
        self.similarity_config_dict = similarity_config_dict
        super().__init__(**kwargs)

    @property
    def type(self):
        return "Data Leakage"

    @property
    def method(self):
        return "Similarity"


class LeakySplits(fob.BrainMethod):
    def initialize(self, samples, brain_key=None):
        return LeakySplitsIndex(samples, self.config, brain_key)


class LeakySplitsIndex(fob.BrainResults):
    def __init__(self, samples, config, brain_key):
        super().__init__(samples, config, brain_key)

        # process arguments to work only with views
        self.split_views = _to_views(
            samples,
            self.config.split_views,
            self.config.split_field,
            self.config.split_tags,
        )
        self._leak_threshold = 0.2
        self._last_computed_threshold = None
        self._leaks = None

        # similarity index setup
        self._similarity_index = None
        self._save_similarity_index = False
        index_found = False
        if self.config.similarity_brain_key is not None:
            # Load similarity brain run if it exists
            if (
                self.config.similarity_brain_key
                in self.samples._dataset.list_brain_runs()
            ):
                self._similarity_index = (
                    self.samples._dataset.load_brain_results(
                        self.config.similarity_brain_key, load_view=True
                    )
                )
                if self._similarity_index is not None:
                    index_found = True
                # check if brain run view lines up with samples provided
                similarity_view = self.samples._dataset.load_brain_view(
                    self.config.similarity_brain_key
                )
                if not set(samples.values("id")) == set(
                    similarity_view.values("id")
                ):
                    raise ValueError(
                        "Provided similarity run doesn't include all samples given. "
                        "Please rerun the similarity with all of the wanted samples."
                    )
            else:
                self._save_similarity_index = True

        # create new similarity index
        if not index_found:
            sim_conf_dict_aux = {
                "name": self.config.similarity_backend,
                "embeddings_field": self.config.embeddings_field,
                "model": self.config.model,
                "model_kwargs": self.config.model_kwargs,
            }
            if self.config.similarity_config_dict is not None:
                # values from args over config dict
                sim_conf_dict_aux = {
                    **self.config.similarity_config_dict,
                    **sim_conf_dict_aux,
                }

            similarity_config_object = sim._parse_config(**sim_conf_dict_aux)
            self._similarity_method = similarity_config_object.build()
            self._similarity_method.ensure_requirements()

            if self._save_similarity_index:
                self._similarity_method.register_run(
                    samples,
                    self.config.similarity_brain_key,
                )

            self._similarity_index = self._similarity_method.initialize(
                samples, self.config.similarity_brain_key
            )

    def set_threshold(self, threshold):
        """Set threshold for leak computation"""
        self._leak_threshold = threshold

    @property
    def leaks(self):
        """
        Returns view with all potential leaks.
        """

        # access cache if possible
        if self._last_computed_threshold == self._leak_threshold:
            return self._leaks

        # populate index if it doesn't have some samples
        if not self._similarity_index.total_index_size == len(self.samples):
            (
                embeddings,
                sample_ids,
                label_ids,
            ) = self._similarity_index.compute_embeddings(self.samples)
            self._similarity_index.add_to_index(
                embeddings, sample_ids, label_ids
            )
        self._similarity_index.find_duplicates(self._leak_threshold)
        duplicates = self._similarity_index.duplicates_view()

        # filter duplicates to just those with neighbors in different splits
        to_keep = []
        for (
            sample_id,
            neighbors,
        ) in self._similarity_index.neighbors_map.items():
            keep_sample = False
            sample_split = self._id2split(sample_id, self.split_views)
            leaks = []
            for n in neighbors:
                if not (
                    self._id2split(n[0], self.split_views) == sample_split
                ):
                    # at least one of the neighbors is from a different split
                    # we keep this one
                    keep_sample = True
                    leaks.append(n[0])

            if keep_sample:
                to_keep.append(sample_id)
                # remove all other samples because they are all from the same split
                to_keep = to_keep + leaks

        duplicates = duplicates.select(to_keep)

        # cache to avoid recomputation
        self._last_computed_threshold = self._leak_threshold
        self._leaks = duplicates

        return duplicates

    @property
    def num_leaks(self):
        """Returns the number of leaks found."""
        return self.leaks.count

    def leaks_by_sample(self, sample):
        """Return view with all leaks related to a certain sample.

        Args:
            sample: sample object or sample id

        """
        # compute leaks if it hasn't happend yet
        if not self._leak_threshold == self._last_computed_threshold:
            leaks = self.leaks

        sample_id = sample if isinstance(sample, str) else sample["id"]

        neighbors = self._similarity_index.neighbors_map[sample_id]
        sample_split = self._id2split(sample_id, self.split_views)
        neighbors_ids = [
            n[0]
            for n in neighbors
            if not self._id2split(n[0], self.split_views) == sample_split
        ]

        return self.samples.select([sample_id] + neighbors_ids)

    def view_without_leaks(self, view):
        return view.exclude(self.leaks.values("id"))

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
        if len(view) < 1:  # no samples in tag
            raise ValueError(
                f"One of the tags provided, '{tag}', has no samples. Make sure every tag has at least one sample."
            )
        views.append(view)

    for i, v in enumerate(views):
        other_tags = [t for j, t in enumerate(tags) if not i == j]
        if len(v.match_tags(other_tags)) > 0:
            raise ValueError(
                f"One or more samples have more than one of the tags provided! Every sample should have at most one of the tags provided."
            )

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
