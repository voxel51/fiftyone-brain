"""
The brains behind FiftyOne: a powerful package for dataset curation, analysis,
and visualization.

See https://github.com/voxel51/fiftyone for more information.

| Copyright 2017-2022, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from .similarity import (
    SimilarityConfig,
    SimilarityResults,
)
from .visualization import (
    VisualizationConfig,
    UMAPVisualizationConfig,
    TSNEVisualizationConfig,
    PCAVisualizationConfig,
    ManualVisualizationConfig,
    VisualizationResults,
)


def compute_hardness(samples, label_field, hardness_field="hardness"):
    """Adds a hardness field to each sample scoring the difficulty that the
    specified label field observed in classifying the sample.

    Hardness is a measure computed based on model prediction output (through
    logits) that summarizes a measure of the uncertainty the model had with the
    sample. This makes hardness quantitative and can be used to detect things
    like hard samples, annotation errors during noisy training, and more.

    All classifications must have their
    :attr:`logits <fiftyone.core.labels.Classification.logits>` attributes
    populated in order to use this method.

    .. note::

        Runs of this method can be referenced later via brain key
        ``hardness_field``.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        label_field: the :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` field to use from
            each sample
        hardness_field ("hardness"): the field name to use to store the
            hardness value for each sample
    """
    import fiftyone.brain.internal.core.hardness as fbh

    return fbh.compute_hardness(samples, label_field, hardness_field)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
    missing_field="possible_missing",
    spurious_field="possible_spurious",
    use_logits=False,
    copy_missing=False,
):
    """Computes the mistakenness of the labels in the specified
    ``label_field``, scoring the chance that the labels are incorrect.

    Mistakenness is computed based on the predictions in the ``pred_field``,
    through either their ``confidence`` or ``logits`` attributes. This measure
    can be used to detect things like annotation errors and unusually hard
    samples.

    This method supports both classifications and detections/polylines.

    For classifications, a ``mistakenness_field`` field is populated on each
    sample that quantifies the likelihood that the label in the ``label_field``
    of that sample is incorrect.

    For detections/polylines, the mistakenness of each object in
    ``label_field`` is computed, using
    :meth:`fiftyone.core.collections.SampleCollection.evaluate_detections` to
    locate corresponding objects in ``pred_field``. Three types of mistakes
    are identified:

    -   **(Mistakes)** Objects in ``label_field`` with a match in
        ``pred_field`` are assigned a mistakenness value in their
        ``mistakenness_field`` that captures the likelihood that the class
        label of the detection in ``label_field`` is a mistake. A
        ``mistakenness_field + "_loc"`` field is also populated that captures
        the likelihood that the detection in ``label_field`` is a mistake due
        to its localization (bounding box).

    -   **(Missing)** Objects in ``pred_field`` with no matches in
        ``label_field`` but which are likely to be correct will have their
        ``missing_field`` attribute set to True. In addition, if
        ``copy_missing`` is True, copies of these objects are *added* to the
        ground truth ``label_field``.

    -   **(Spurious)** Objects in ``label_field`` with no matches in
        ``pred_field`` but which are likely to be incorrect will have their
        ``spurious_field`` attribute set to True.

    In addition, for detections/polylines, the following sample-level fields
    are populated:

    -   **(Mistakes)** The ``mistakenness_field`` of each sample is populated
        with the maximum mistakenness of the objects in ``label_field``

    -   **(Missing)** The ``missing_field`` of each sample is populated with
        the number of missing objects that were deemed missing from
        ``label_field``.

    -   **(Spurious)** The ``spurious_field`` of each sample is populated with
        the number of objects in ``label_field`` that were given deemed
        spurious.

    .. note::

        Runs of this method can be referenced later via brain key
        ``mistakenness_field``.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        pred_field: the name of the predicted label field to use from each
            sample. Can be of type
            :class:`fiftyone.core.labels.Classification`,
            :class:`fiftyone.core.labels.Classifications`, or
            :class:`fiftyone.core.labels.Detections`
        label_field ("ground_truth"): the name of the "ground truth" label
            field that you want to test for mistakes with respect to the
            predictions in ``pred_field``. Must have the same type as
            ``pred_field``
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        missing_field ("possible_missing): the field in which to store
            per-sample counts of potential missing detections/polylines
        spurious_field ("possible_spurious): the field in which to store
            per-sample counts of potential spurious detections/polylines
        use_logits (False): whether to use logits (True) or confidence (False)
            to compute mistakenness. Logits typically yield better results,
            when they are available
        copy_missing (False): whether to copy predicted detections/polylines
            that were deemed to be missing into ``label_field``
    """
    import fiftyone.brain.internal.core.mistakenness as fbm

    return fbm.compute_mistakenness(
        samples,
        pred_field,
        label_field,
        mistakenness_field,
        missing_field,
        spurious_field,
        use_logits,
        copy_missing,
    )


def compute_uniqueness(
    samples,
    uniqueness_field="uniqueness",
    roi_field=None,
    embeddings=None,
    model=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    If no ``embeddings`` or ``model`` is provided, a default model is used to
    generate embeddings.

    .. note::

        Runs of this method can be referenced later via brain key
        ``uniqueness_field``.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        uniqueness_field ("uniqueness"): the field name to use to store the
            uniqueness value for each sample
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute uniqueness
        embeddings (None): pre-computed embeddings to use. Can be any of the
            following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``roi_field`` is specified,  a dict mapping sample IDs to
                ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://voxel51.com/docs/fiftyone/user_guide/model_zoo/models.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 1.1`` to expand the boxes by 10%, and set ``alpha = 0.9``
            to contract the boxes by 10%. Only applicable when a ``model`` and
            ``roi_field`` are specified
        batch_size (None): a batch size to use when computing embeddings. Only
            applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
    """
    import fiftyone.brain.internal.core.uniqueness as fbu

    return fbu.compute_uniqueness(
        samples,
        uniqueness_field,
        roi_field,
        embeddings,
        model,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
    )


def compute_visualization(
    samples,
    patches_field=None,
    embeddings=None,
    points=None,
    brain_key=None,
    num_dims=2,
    method="umap",
    model=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    **kwargs,
):
    """Computes a low-dimensional representation of the samples' media or their
    patches that can be interactively visualized.

    The representation can be visualized by calling the
    :meth:`sort_by_similarity() <fiftyone.brain.similarity.SimilarityResults.sort_by_similarity>`
    method of the returned
    :class:`fiftyone.brain.visualization.VisualizationResults` object.

    If no ``embeddings`` or ``model`` is provided, a default model is used to
    generate embeddings.

    You can use the ``method`` parameter to select the dimensionality-reduction
    method to use, and you can optionally customize the method by passing
    additional parameters for the method's
    :class:`fiftyone.brain.visualization.VisualizationConfig` class as
    ``kwargs``.

    The supported ``method`` values and their associated config classes are:

    -   ``"umap"``: :class:`fiftyone.brain.visualization.UMAPVisualizationConfig`
    -   ``"tsne"``: :class:`fiftyone.brain.visualization.TSNEVisualizationConfig`
    -   ``"pca"``: :class:`fiftyone.brain.visualization.PCAVisualizationConfig`
    -   ``"manual"``: :class:`fiftyone.brain.visualization.ManualVisualizationConfig`

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        patches_field (None): a sample field defining the image patches in each
            sample that have been/will be embedded. Must be of type
            :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines`
        embeddings (None): pre-computed embeddings to use. Can be any of the
            following:

            -   a ``num_samples x num_embedding_dims`` array of embeddings
            -   if ``patches_field`` is specified,  a dict mapping sample IDs
                to ``num_patches x num_embedding_dims`` arrays of patch
                embeddings
            -   the name of a dataset field containing the embeddings to use
        points (None): a pre-computed low-dimensional representation to use. If
            provided, no embeddings will be computed. Can be any of the
            following:

            -   a ``num_samples x num_dims`` array of points
            -   if ``patches_field`` is specified, a ``num_patches x num_dims``
                array of points whose rows correspond to the flattened list of
                patches whose IDs are shown below::

                    # The list of patch IDs that the rows of `points` must match
                    _, id_field = samples._get_label_field_path(patches_field, "id")
                    patch_ids = samples.values(id_field, unwind=True)

        brain_key (None): a brain key under which to store the results of this
            method
        num_dims (2): the dimension of the visualization space
        method ("umap"): the dimensionality-reduction method to use. Supported
            values are ``("umap", "tsne", "pca", "manual")``
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://voxel51.com/docs/fiftyone/user_guide/model_zoo/index.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``patches_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 1.1`` to expand the boxes by 10%, and set ``alpha = 0.9``
            to contract the boxes by 10%. Only applicable when a ``model`` and
            ``patches_field`` are specified
        batch_size (None): an optional batch size to use when computing
            embeddings. Only applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        **kwargs: optional keyword arguments for the constructor of the
            :class:`fiftyone.brain.visualization.VisualizationConfig`
            being used

    Returns:
        a :class:`fiftyone.brain.visualization.VisualizationResults`
    """
    import fiftyone.brain.internal.core.visualization as fbv

    return fbv.compute_visualization(
        samples,
        patches_field,
        embeddings,
        points,
        brain_key,
        num_dims,
        method,
        model,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
        **kwargs,
    )


def compute_similarity(
    samples,
    patches_field=None,
    embeddings=None,
    brain_key=None,
    metric="euclidean",
    model=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
):
    """Uses embeddings to index the samples or their patches so that you can
    query/sort by visual similarity.

    Calling this method (or loading existing results) only generates the index.
    You can then call the methods exposed on the retuned
    :class:`fiftyone.brain.similarity.SimilarityResults` object to perform the
    following operations:

    -   :meth:`sort_by_similarity() <fiftyone.brain.similarity.SimilarityResults.sort_by_similarity>`:
        Sort the samples in the collection by visual similarity to a specific
        example or example(s)

    -   :meth:`find_duplicates() <fiftyone.brain.similarity.SimilarityResults.find_duplicates>`:
        Query the index to find all examples with near-duplicates in the
        collection

    -   :meth:`find_unique() <fiftyone.brain.similarity.SimilarityResults.find_unique>`:
        Query the index to select a subset of examples of a specified size that
        are maximally unique with respect to each other

    If no ``embeddings`` or ``model`` is provided, a default model is used to
    generate embeddings.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        patches_field (None): a sample field defining the image patches in each
            sample that have been/will be embedded. Must be of type
            :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines`
        embeddings (None): pre-computed embeddings to use. Can be any of the
            following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``patches_field`` is specified,  a dict mapping sample IDs
                to ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

        brain_key (None): a brain key under which to store the results of this
            method
        metric ("euclidean"): the embedding distance metric to use. See
            ``sklearn.metrics.pairwise_distance`` for supported values
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://voxel51.com/docs/fiftyone/user_guide/model_zoo/index.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``patches_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 1.1`` to expand the boxes by 10%, and set ``alpha = 0.9``
            to contract the boxes by 10%. Only applicable when a ``model`` and
            ``patches_field`` are specified
        batch_size (None): an optional batch size to use when computing
            embeddings. Only applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample

    Returns:
        a :class:`fiftyone.brain.similarity.SimilarityResults`
    """
    import fiftyone.brain.internal.core.similarity as fbs

    return fbs.compute_similarity(
        samples,
        patches_field,
        embeddings,
        brain_key,
        metric,
        model,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
    )


def compute_exact_duplicates(samples, num_workers=None, skip_failures=True):
    """Detects duplicate media in a sample collection.

    This method detects exact duplicates with the same filehash. Use
    :meth:`compute_similarity` to detect near-duplicate images.

    If duplicates are found, the first instance in ``samples`` will be the key
    in the returned dictionary, while the subsequent duplicates will be the
    values in the corresponding list.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        num_workers (None): an optional number of processes to use
        skip_failures (True): whether to gracefully ignore samples whose
            filehash cannot be computed

    Returns:
        a dictionary mapping IDs of samples with exact duplicates to lists of
        IDs of the duplicates for the corresponding sample
    """
    import fiftyone.brain.internal.core.duplicates as fbd

    return fbd.compute_exact_duplicates(samples, num_workers, skip_failures)
