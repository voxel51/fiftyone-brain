"""
Uniqueness methods.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import deepcopy
import logging

import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.validation as fov

import fiftyone.brain.internal.core.utils as fbu
import fiftyone.brain.internal.models as fbm


logger = logging.getLogger(__name__)

_ALLOWED_ROI_FIELD_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.Polyline,
    fol.Polylines,
)

_DEFAULT_MODEL = "simple-resnet-cifar10"
_DEFAULT_BATCH_SIZE = 16


def compute_uniqueness(
    samples,
    uniqueness_field,
    roi_field,
    embeddings,
    model,
    model_kwargs,
    force_square,
    alpha,
    batch_size,
    num_workers,
    skip_failures,
    progress,
    similarity_backend=None,
    similarity_index=None,
):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Uniqueness is computed based on a classification model.  Each sample is
    # embedded into a vector space based on the model. Then, we compute the
    # knn's (k is a parameter of the uniqueness function). The uniqueness is
    # then proportional to these distances. The intuition is that a sample is
    # unique when it is far from other samples in the set. This is different
    # than, say, "representativeness" which would stress samples that are core
    # to dense clusters of related samples.
    #
    if similarity_backend and similarity_index:
        raise IOError(
            "At least one of (similarity_backend, similarity_index) values need to be None."
        )

    if not (similarity_index or similarity_backend):
        similarity_backend = fb.brain_config.default_similarity_backend

    fov.validate_image_collection(samples)

    if roi_field is not None:
        fov.validate_collection_label_fields(
            samples, roi_field, _ALLOWED_ROI_FIELD_TYPES
        )

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings,
            patches_field=roi_field,
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if model is None and embeddings is None and not embeddings_exist:
        model = fbm.load_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = UniquenessConfig(
        uniqueness_field,
        similarity_backend=similarity_backend,
        roi_field=roi_field,
        embeddings_field=embeddings_field,
        model=model,
        model_kwargs=model_kwargs,
    )
    brain_key = uniqueness_field
    brain_method = config.build()
    brain_method.ensure_requirements()
    brain_method.register_run(samples, brain_key, cleanup=False)

    if roi_field is not None:
        # @todo experiment with mean(), max(), abs().max(), etc
        agg_fcn = lambda e: np.mean(e, axis=0)
    else:
        agg_fcn = None

    if similarity_index:
        sample_ids = similarity_index.sample_ids
        embeddings = np.array(
            similarity_index.get_embeddings(sample_ids=sample_ids)[0]
        )
    elif config.similarity_method:
        embeddings, sample_ids, _ = fbu.get_embeddings(
            samples,
            model=model,
            model_kwargs=model_kwargs,
            patches_field=roi_field,
            embeddings_field=embeddings_field,
            embeddings=embeddings,
            force_square=force_square,
            alpha=alpha,
            handle_missing="image",
            agg_fcn=agg_fcn,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=progress,
        )
        brain_method = config.similarity_method.build()
        similarity_index = brain_method.initialize(
            samples=samples, brain_key=f"{similarity_backend}_index"
        )
        similarity_index.add_to_index(embeddings, sample_ids)
    else:
        raise AssertionError("Similarity index not available.")

    logger.info("Computing uniqueness...")
    uniqueness = _compute_uniqueness(embeddings, similarity_index)

    # Ensure field exists, even if `uniqueness` is empty
    samples._dataset.add_sample_field(uniqueness_field, fof.FloatField)

    uniqueness = {_id: u for _id, u in zip(sample_ids, uniqueness)}
    if uniqueness:
        samples.set_values(uniqueness_field, uniqueness, key_field="id")

    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Uniqueness computation complete")


def _compute_uniqueness(
    embeddings, similarity_index, metric="euclidean", n_neighbors=3
):
    num_embeddings = len(embeddings)
    if num_embeddings <= n_neighbors:
        return [1] * num_embeddings

    _, dists_list = similarity_index._kneighbors(
        query=embeddings, k=n_neighbors + 1, return_dists=True
    )
    dists = np.array(dists_list)

    # @todo experiment on which method for assessing uniqueness is best
    #
    # To get something going, for now, just take a weighted mean
    #
    weights = [0.6, 0.3, 0.1]
    sample_dists = np.mean(dists[:, 1:] * weights, axis=1)

    # Normalize to keep the user on common footing across datasets
    sample_dists /= sample_dists.max()

    return sample_dists


# @todo move to `fiftyone/brain/uniqueness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class UniquenessConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        uniqueness_field,
        roi_field=None,
        embeddings_field=None,
        model=None,
        model_kwargs=None,
        similarity_backend=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.uniqueness_field = uniqueness_field
        self.roi_field = roi_field
        self.embeddings_field = embeddings_field
        self.model = model
        self.model_kwargs = model_kwargs

        # Similarity backend.
        self.similarity_method = None
        if similarity_backend:
            backends = fb.brain_config.similarity_backends
            if similarity_backend not in backends:
                raise ValueError(
                    f"Unsupported backend {similarity_backend}. The available backends are {sorted(backends.keys())}"
                )
            backend_params = deepcopy(backends[similarity_backend])
            config_cls = kwargs.pop("config_cls", None)
            if config_cls is None:
                config_cls = backend_params.pop("config_cls", None)
            if config_cls is None:
                raise ValueError(
                    f"Similarity backend {similarity_backend} has no `config_cls`"
                )
            if etau.is_str(config_cls):
                config_cls = etau.get_class(config_cls)
            backend_params.update(**kwargs)
            self.similarity_method = config_cls(**backend_params)

        super().__init__(**kwargs)

    @property
    def type(self):
        return "uniqueness"

    @property
    def method(self):
        return "neighbors"


class Uniqueness(fob.BrainMethod):
    def ensure_requirements(self):
        pass

    def get_fields(self, samples, brain_key):
        fields = [self.config.uniqueness_field]
        if self.config.roi_field is not None:
            fields.append(self.config.roi_field)

        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields

    def cleanup(self, samples, brain_key):
        uniqueness_field = self.config.uniqueness_field
        samples._dataset.delete_sample_fields(uniqueness_field, error_level=1)

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(
            brain_key, "uniqueness_field", existing_info
        )
