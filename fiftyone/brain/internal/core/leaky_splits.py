"""
Finds leaks between splits.
"""

from copy import copy
import warnings

from fiftyone import ViewField as F
import fiftyone.core.brain as fob
import fiftyone.brain.similarity as sim
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")

_DEFAULT_MODEL = "resnet18-imagenet-torch"


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
    """See ``fiftyone/brain/__init__.py``."""

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

    if results._save_similarity_index:
        results._similarity_method.register_run(
            samples,
            results.config.similarity_brain_key,
        )

    results.set_threshold(threshold)
    leaks = results.leaks_view()

    if brain_key is not None:
        brain_method.save_run_results(samples, brain_key, results)
    if results._save_similarity_index:
        results._similarity_method.save_run_results(
            samples, similarity_brain_key, results._similarity_index
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
        return "leakage"

    @property
    def method(self):
        return "similarity"


class LeakySplits(fob.BrainMethod):
    def initialize(self, samples, brain_key=None):
        self.index = LeakySplitsIndex(samples, self.config, brain_key)
        return self.index

    def cleanup(self, samples, brain_key):
        self.index._similarity_method.cleanup(samples, brain_key)
        super.cleanup(samples, brain_key)


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

        total_len_views = sum([len(v) for v in self.split_views.values()])
        if len(samples) > total_len_views:
            warnings.warn(
                "More samples passed than samples in splits. These will not be"
                " considered for leak computation."
            )

        if total_len_views > len(samples):
            raise ValueError(
                "Found more items in splits than in total samples!\n"
                "Splits are supposed to be a disjoint cover of the samples."
            )

        self._id2split = self._id2splitConstructor()
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
                index_found = True
                self._similarity_index = (
                    self.samples._dataset.load_brain_results(
                        self.config.similarity_brain_key, load_view=True
                    )
                )

                if not len(self._similarity_index._samples) == len(samples):
                    warnings.warn(
                        "Passed similarity index is not of the same size as the samples passed. "
                        "This can cause errors. Make sure the similarity index passed has all "
                        "of the samples needed "
                    )
            else:
                self._save_similarity_index = True

        # create new similarity index
        if not index_found:
            sim_conf_dict_kwargs = {}
            # if arguments aren't None they take precedence
            if self.config.similarity_backend is not None:
                sim_conf_dict_kwargs["name"] = self.config.similarity_backend
            if self.config.embeddings_field is not None:
                sim_conf_dict_kwargs[
                    "embeddings_field"
                ] = self.config.embeddings_field
            if self.config.model is not None:
                sim_conf_dict_kwargs["model"] = self.config.model
            if self.config.model_kwargs is not None:
                sim_conf_dict_kwargs["model_kwargs"] = self.config.model_kwargs

            # empty conf if conf dict isn't provided
            similarity_conf_provided = {}
            if self.config.similarity_config_dict is not None:
                similarity_conf_provided = self.config.similarity_config_dict
            sim_conf_dict_kwargs = {
                "name": None,  # default if user doesn't provide any value
                "model": _DEFAULT_MODEL,  # default if user doesn't provide any value
                **similarity_conf_provided,  # conf over defaults
                **sim_conf_dict_kwargs,  # arguments over conf
            }

            backend_name = sim_conf_dict_kwargs.pop("name")

            similarity_config_object = sim._parse_config(
                backend_name, **sim_conf_dict_kwargs
            )
            self._similarity_method = similarity_config_object.build()
            self._similarity_method.ensure_requirements()

            self._similarity_index = self._similarity_method.initialize(
                samples, self.config.similarity_brain_key
            )

    def set_threshold(self, threshold):
        """Set threshold for leak computation"""
        self._leak_threshold = threshold

    def leaks_view(self):
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

        # check if duplicates already computed in the index
        if not self._similarity_index.thresh == self._leak_threshold:
            self._similarity_index.find_duplicates(self._leak_threshold)
        duplicates = self._similarity_index.duplicates_view()

        # filter duplicates to just those with neighbors in different splits
        to_keep = []
        for (
            sample_id,
            neighbors,
        ) in self._similarity_index.neighbors_map.items():
            keep_sample = False
            sample_split = self._id2split.get(sample_id, None)
            if sample_split is None:
                _throw_index_is_bigger_warning(sample_id)
                continue  # sample is in index but not passed to leaky_splits
            leaks = []
            for n in neighbors:
                neighbor_split = self._id2split.get(n[0], None)
                if neighbor_split is None:
                    _throw_index_is_bigger_warning(n[0])
                    continue
                if not (neighbor_split == sample_split):
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

    def leaks_for_sample(self, sample):
        """Return view with all leaks related to a certain sample.

        Args:
            sample: sample object or sample id

        """
        # compute leaks if it hasn't happend yet
        if not self._leak_threshold == self._last_computed_threshold:
            _ = self.leaks_view()

        sample_id = sample if isinstance(sample, str) else sample["id"]
        sample_split = self._id2split[sample_id]
        neighbors_ids = []
        if sample_id in self._similarity_index.neighbors_map.keys():
            neighbors = self._similarity_index.neighbors_map[sample_id]
            neighbors_ids = [
                n[0]
                for n in neighbors
                if not self._id2split[n[0]] == sample_split
            ]
        else:
            for (
                unique_id,
                neighbors,
            ) in self._similarity_index.neighbors_map.items():
                if sample_id in [n[0] for n in neighbors]:
                    neighbors_ids = [
                        n[0]
                        for n in neighbors
                        if not self._id2split[n[0]] == sample_split
                    ]
                    neighbors_ids.append(unique_id)
                    break

        return self.samples.select([sample_id] + neighbors_ids)

    def no_leaks_view(self, view):
        return view.exclude(self.leaks_view().values("id"))

    def tag_leaks(self, tag="leak"):
        """Tag leaks"""
        for s in self.leaks_view().iter_samples():
            s.tags.append(tag)
            s.save()

    def _id2splitConstructor(self):

        # do this once at the beggining of the run
        # has O(n) memory cost but memory is cheap compared to
        # doing a couple of `in` operations per sample every run
        # I want that sweet sweet O(1) lookup
        id2split = {}
        for split_name, split_view in self.split_views.items():
            sample_ids = split_view.values("id")
            id2split.update({sid: split_name for sid in sample_ids})

        return id2split


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

    views = {}
    for val in field_values:
        view = samples.match(F(field) == val)
        views[val] = view

    return views


def _tags_to_views(samples, tags):
    if len(tags) < 2:
        raise ValueError("Must provide at least two tags.")

    views = {}
    for tag in tags:
        view = samples.match_tags([tag])
        if len(view) < 1:  # no samples in tag
            raise ValueError(
                f"One of the tags provided, '{tag}', has no samples. Make sure every tag has at least one sample."
            )
        views[tag] = view

    for tag, view in views.items():
        other_tags = [t for t in tags if not t == tag]
        if len(view.match_tags(other_tags)) > 0:
            raise ValueError(
                f"One or more samples have more than one of the tags provided! Every sample should have at most one of the tags provided."
            )

    return views


def _throw_index_is_bigger_warning(sample_id):
    warnings.warn(
        f"Tried querying sample with id {sample_id}. This sample is not in any split\n"
        "This sample will not be considered for leaks!"
    )
