"""
Finds leaks between splits.
"""

from copy import copy

import fiftyone as fo

import fiftyone.core.brain as fob
import fiftyone.brain.similarity as sim
import fiftyone.brain.internal.core.sklearn as skl_sim


def compute_leaky_splits(
    samples,
    split_tags,
    method="similarity",
    similarity_backend=None,
    similarity_backend_kwargs=None,
    **kwargs,
):
    print("bar")


### fancy general implementation. Not worth time currently.
class LeakySplitsConfig(fob.BrainMethodConfig):
    """Configuration for the leaky split BrainMethod

    Args:
        split_tags: list of str, corresponding to the tags of the splits
        similarity_backend_config: :class:`fiftyone.core.brain.SimilarityConfig` and
            :class:`fiftyone.core.brain.DuplicatesMixin`

        Instead of sending an actual config instance it may be better to send kwargs
        or something else. This can be implmented later.
    """

    def __init__(self, split_tags, similarity_backend_config, **kwargs):
        self.split_tags = split_tags
        self.similarity_backend_config = similarity_backend_config

        super().__init__()

    @property
    def type(self):
        return "leaky splits"


class LeakySplits(fob.BrainMethod):
    """LeakySplits class for finding leaks between different different splits.

    Args:
        config: a :class:`LeakySplitConfig`
    """

    def _validate_run(self, samples, key, existing_info):

        # check enough tags
        split_tags = self.config.split_tags

        if len(split_tags) < 2:
            raise ValueError(
                "Please include at least two tags in least so that they can be compared!"
            )

        sample_tag_counts = samples.count_sample_tags()

        # validate tags are in collection
        for tag in split_tags:
            if tag not in sample_tag_counts.keys():
                raise ValueError(
                    f"Given tag {tag} but it doesn't exist in collection, choose from: "
                    f"{sample_tag_counts.keys()}"
                )

            # may be redudnant if tags are deleted if they have no use but keeping in
            # just in case
            if sample_tag_counts[tag] == 0:
                raise ValueError(
                    f"The tag {tag} has no samples! Make sure each tag has at least 1"
                    " sample associated with it."
                )


class LeakySplitsIndex(fob.BrainResults):
    """Class for storing LeakySplits data.

    Args:
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)

        self._sim_config = config.similarity_backend_config
        sim_instance = self._sim_config.build()
        self._sim_backend_index = sim_instance.initialize(samples, brain_key)

    def find_all_potential_leaks(self):
        """Returns a view with samples that are most likely to be leaks"""
        return self._sim_backend_index.view

    def sort_by_leak_potential(
        self,
        query,
        tags_of_interest,
    ):
        return self._sim_backend_index.view


###


### ugly fast implemntation so that I can start testing interesting things.

_BASIC_METHODS = ["exact", "neural"]


class LeakySplitsSKLConfig(skl_sim.SklearnSimilarityConfig):
    def __init__(self, split_tags, method, **kwargs):
        self.split_tags = split_tags
        self._method = method
        super().__init__(**kwargs)

    @property
    def method(self):
        return self._method


class LeakySplitsSKL(skl_sim.SklearnSimilarity):
    def initialize(self, samples, brain_key):
        return LeakySplitsSKLIndex(
            samples, self.config, brain_key, backend=self
        )


class LeakySplitsSKLIndex(skl_sim.SklearnSimilarityIndex):
    def __init__(self, samples, config, brain_key, **kwargs):
        super().__init__(samples, config, brain_key, **kwargs)

    def sort_by_leak_potential(self, sample, k=None, dist_field=None):

        # isolate view to search through
        sample_split = set(self.config.split_tags) & set(sample.tags)
        if len(sample_split) > 1:
            raise ValueError("sample belongs to multiple splits.")
        sample_split = sample_split.pop()

        tags_to_search = copy(self.config.split_tags)
        tags_to_search.remove(sample_split)

        self.use_view(self._dataset.match_tags(tags_to_search))

        # run similarity
        return self.sort_by_similarity(sample["id"], k, dist_field=dist_field)


###
