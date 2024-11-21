"""
Finds leaks between splits.
"""

from copy import copy

from fiftyone import ViewField as F
import fiftyone.core.brain as fob
import fiftyone.brain.similarity as sim
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov

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
    leaks = results.leaks

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
        self.id2split = self._id2splitConstructor()
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
                # check if brain run view lines up with samples provided
                # TODO: Brian says there is a better way of doing this, need to think of how
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
            sim_conf_dict_kwargs = {
                "name": self.config.similarity_backend,
                "embeddings_field": self.config.embeddings_field,
                "model": self.config.model,
                "model_kwargs": self.config.model_kwargs,
            }
            if self.config.similarity_config_dict is not None:
                # values from args over config dict
                sim_conf_dict_kwargs = {
                    **self.config.similarity_config_dict,
                    **sim_conf_dict_kwargs,
                }

            similarity_config_object = sim._parse_config(
                **sim_conf_dict_kwargs
            )
            self._similarity_method = similarity_config_object.build()
            self._similarity_method.ensure_requirements()

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
            sample_split = self.id2split[sample_id]
            leaks = []
            for n in neighbors:
                if not (self.id2split[n[0]] == sample_split):
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
        sample_split = self.id2split[sample_id]
        neighbors_ids = []
        if sample_id in self._similarity_index.neighbors_map.keys():
            neighbors = self._similarity_index.neighbors_map[sample_id]
            neighbors_ids = [
                n[0]
                for n in neighbors
                if not self.id2split[n[0]] == sample_split
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
                        if not self.id2split[n[0]] == sample_split
                    ]
                    neighbors_ids.append(unique_id)
                    break

        return self.samples.select([sample_id] + neighbors_ids)

    def view_without_leaks(self, view):
        return view.exclude(self.leaks.values("id"))

    def tag_leaks(self, tag="leak"):
        """Tag leaks"""
        for s in self.leaks.iter_samples():
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

    def cleanup(self):
        self._similarity_index.cleanup()


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

    for i, v in enumerate(views):
        other_tags = [t for j, t in enumerate(tags) if not i == j]
        if len(v.match_tags(other_tags)) > 0:
            raise ValueError(
                f"One or more samples have more than one of the tags provided! Every sample should have at most one of the tags provided."
            )

    return views
