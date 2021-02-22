"""
Methods that compute insights related to sample hardness.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications)


def compute_hardness(samples, label_field, hardness_field):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Hardness is computed directly as the entropy of the logits
    #

    fov.validate_collection(samples)
    fov.validate_collection_label_fields(samples, label_field, _ALLOWED_TYPES)

    if samples._is_frame_field(label_field):
        raise ValueError("Hardness does not yet support frame fields")

    config = HardnessConfig(label_field, hardness_field)
    brain_key = hardness_field
    brain_method = config.build()
    brain_method.register_run(samples, brain_key)

    samples = samples.select_fields(label_field)

    logger.info("Computing hardness...")
    with fou.ProgressBar() as pb:
        for sample in pb(samples):
            label = _get_data(sample, label_field)
            hardness = entropy(softmax(np.asarray(label.logits)))
            sample[hardness_field] = hardness
            sample.save()


class HardnessConfig(fob.BrainMethodConfig):
    def __init__(self, label_field, hardness_field, **kwargs):
        super().__init__(**kwargs)
        self.label_field = label_field
        self.hardness_field = hardness_field

    @property
    def method(self):
        return "hardness"


class Hardness(fob.BrainMethod):
    def get_fields(self, samples, brain_key):
        return [self.config.label_field, self.config.hardness_field]

    def cleanup(self, samples, brain_key):
        hardness_field = self.config.hardness_field
        samples._dataset.delete_sample_fields(hardness_field)

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(brain_key, "hardness_field", existing_info)


def _get_data(sample, label_field):
    label = fov.get_field(
        sample, label_field, allowed_types=_ALLOWED_TYPES, allow_none=False
    )

    if label.logits is None:
        raise ValueError(
            "Sample '%s' field '%s' has no logits" % (sample.id, label_field)
        )

    return label
