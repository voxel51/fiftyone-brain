"""
Finds leaks between splits.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import eta.core.utils as etau

import fiftyone.core.validation as fov

import fiftyone.brain as fb
import fiftyone.brain.similarity as fbs
import fiftyone.brain.internal.core.utils as fbu


_DEFAULT_MODEL = "resnet18-imagenet-torch"


def compute_leaky_splits(
    samples,
    splits=None,
    threshold=None,
    embeddings=None,
    similarity_index=None,
    model=None,
    model_kwargs=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    progress=None,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_image_collection(samples)

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings,
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
        model = _DEFAULT_MODEL

    if similarity_index is None:
        similarity_index = fb.compute_similarity(
            samples,
            backend="sklearn",
            model=model,
            model_kwargs=model_kwargs,
            embeddings=embeddings or embeddings_field,
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

    similarity_index.find_leaks(splits, threshold)

    return similarity_index
