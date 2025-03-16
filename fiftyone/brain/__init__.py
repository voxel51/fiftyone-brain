"""
The brains behind FiftyOne: a powerful package for dataset curation, analysis,
and visualization.

See https://github.com/voxel51/fiftyone for more information.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone.brain.config as _foc

from .similarity import (
    Similarity,
    SimilarityConfig,
    SimilarityIndex,
)
from .visualization import (
    Visualization,
    VisualizationConfig,
    VisualizationResults,
)


brain_config = _foc.load_brain_config()


def compute_hardness(
    samples,
    label_field,
    hardness_field="hardness",
    progress=None,
):
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
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
    """
    import fiftyone.brain.internal.core.hardness as fbh

    return fbh.compute_hardness(samples, label_field, hardness_field, progress)


def compute_mistakenness(
    samples,
    pred_field,
    label_field,
    mistakenness_field="mistakenness",
    missing_field="possible_missing",
    spurious_field="possible_spurious",
    use_logits=False,
    copy_missing=False,
    progress=None,
):
    """Computes the mistakenness (likelihood of being incorrect) of the labels
    in ``label_field`` based on the predcted labels in ``pred_field``.

    Mistakenness is measured based on either the ``confidence`` or ``logits``
    of the predictions in ``pred_field``. This measure can be used to detect
    things like annotation errors and unusually hard samples.

    For classifications, a ``mistakenness_field`` field is populated on each
    sample that quantifies the likelihood that the label in the ``label_field``
    of that sample is incorrect.

    For objects (detections, polylines, keypoints, etc), the mistakenness of
    each object in ``label_field`` is computed, using
    :meth:`fiftyone.core.collections.SampleCollection.evaluate_detections` to
    locate corresponding objects in ``pred_field``. Three types of mistakes
    are identified:

    -   **(Mistakes)** Objects in ``label_field`` with a match in
        ``pred_field`` are assigned a mistakenness value in their
        ``mistakenness_field`` that captures the likelihood that the class
        label of the object in ``label_field`` is a mistake. A
        ``mistakenness_field + "_loc"`` field is also populated that captures
        the likelihood that the object in ``label_field`` is a mistake due
        to its localization (bounding box).

    -   **(Missing)** Objects in ``pred_field`` with no matches in
        ``label_field`` but which are likely to be correct will have their
        ``missing_field`` attribute set to True. In addition, if
        ``copy_missing`` is True, copies of these objects are *added* to the
        ground truth ``label_field``.

    -   **(Spurious)** Objects in ``label_field`` with no matches in
        ``pred_field`` but which are likely to be incorrect will have their
        ``spurious_field`` attribute set to True.

    In addition, for objects, the following sample-level fields are populated:

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
            :class:`fiftyone.core.labels.Classifications`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polylines`,
            :class:`fiftyone.core.labels.Keypoints`, or
            :class:`fiftyone.core.labels.TemporalDetections`
        label_field: the name of the "ground truth" label field that you want
            to test for mistakes with respect to the predictions in
            ``pred_field``. Must have the same type as ``pred_field``
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        missing_field ("possible_missing): the field in which to store
            per-sample counts of potential missing objects
        spurious_field ("possible_spurious): the field in which to store
            per-sample counts of potential spurious objects
        use_logits (False): whether to use logits (True) or confidence (False)
            to compute mistakenness. Logits typically yield better results,
            when they are available
        copy_missing (False): whether to copy predicted objects that were
            deemed to be missing into ``label_field``
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
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
        progress,
    )


def compute_uniqueness(
    samples,
    uniqueness_field="uniqueness",
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
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    If no ``embeddings``, ``similarity_index``, or ``model`` is provided, a
    default model is used to generate embeddings.

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
        embeddings (None): if no ``model`` is provided, this argument specifies
            pre-computed embeddings to use, which can be any of the following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``roi_field`` is specified,  a dict mapping sample IDs to
                ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

            If a ``model`` is provided, this argument specifies the name of a
            field in which to store the computed embeddings. In either case,
            when working with patch embeddings, you can provide either the
            fully-qualified path to the patch embeddings or just the name of
            the label attribute in ``roi_field``
        similarity_index (None): a
            :class:`fiftyone.brain.similarity.SimilarityIndex` or the brain key
            of a similarity index to use to load pre-computed embeddings
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/models.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``roi_field`` are specified
        batch_size (None): a batch size to use when computing embeddings. Only
            applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
    """
    import fiftyone.brain.internal.core.uniqueness as fbu

    return fbu.compute_uniqueness(
        samples,
        uniqueness_field,
        roi_field,
        embeddings,
        similarity_index,
        model,
        model_kwargs,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
        progress,
    )


def compute_representativeness(
    samples,
    representativeness_field="representativeness",
    method="cluster-center",
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
    """Adds a representativeness field to each sample scoring how representative
    of nearby samples it is.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    If no ``embeddings``, ``similarity_index``, or ``model`` is provided, a
    default model is used to generate embeddings.

    .. note::

        Runs of this method can be referenced later via brain key
        ``representativeness_field``.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        representativeness_field ("representativeness"): the field name to use
            to store the representativeness value for each sample
        method ("cluster-center"): the name of the method to use to compute the
            representativeness. The supported values are
            ``["cluster-center", 'cluster-center-downweight']``.
            ``"cluster-center"` will make a sample's representativeness
            proportional to it's proximity to cluster centers, while
            ``"cluster-center-downweight"`` will ensure more diversity in
            representative samples
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute representativeness
        embeddings (None): if no ``model`` is provided, this argument specifies
            pre-computed embeddings to use, which can be any of the following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``roi_field`` is specified,  a dict mapping sample IDs to
                ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

            If a ``model`` is provided, this argument specifies the name of a
            field in which to store the computed embeddings. In either case,
            when working with patch embeddings, you can provide either the
            fully-qualified path to the patch embeddings or just the name of
            the label attribute in ``roi_field``
        similarity_index (None): a
            :class:`fiftyone.brain.similarity.SimilarityIndex` or the brain key
            of a similarity index to use to load pre-computed embeddings
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/models.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``roi_field`` are specified
        batch_size (None): a batch size to use when computing embeddings. Only
            applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
    """
    import fiftyone.brain.internal.core.representativeness as fbr

    return fbr.compute_representativeness(
        samples,
        representativeness_field,
        method,
        roi_field,
        embeddings,
        similarity_index,
        model,
        model_kwargs,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
        progress,
    )


def compute_visualization(
    samples,
    patches_field=None,
    embeddings=None,
    points=None,
    create_index=False,
    points_field=None,
    brain_key=None,
    num_dims=2,
    method=None,
    similarity_index=None,
    model=None,
    model_kwargs=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    progress=None,
    **kwargs,
):
    """Computes a low-dimensional representation of the samples' media or their
    patches that can be interactively visualized.

    The representation can be visualized by calling the
    :meth:`visualize() <fiftyone.brain.visualization.VisualizationResults.visualize>`
    method of the returned
    :class:`fiftyone.brain.visualization.VisualizationResults` object.

    If no ``embeddings``, ``similarity_index``, or ``model`` is provided, a
    default model is used to generate embeddings.

    You can use the ``method`` parameter to select the dimensionality reduction
    method to use, and you can optionally customize the method by passing
    additional parameters for the method's
    :class:`fiftyone.brain.visualization.VisualizationConfig` class as
    ``kwargs``.

    The builtin ``method`` values and their associated config classes are:

    -   ``"umap"``: :class:`fiftyone.brain.visualization.UMAPVisualizationConfig`
    -   ``"tsne"``: :class:`fiftyone.brain.visualization.TSNEVisualizationConfig`
    -   ``"pca"``: :class:`fiftyone.brain.visualization.PCAVisualizationConfig`
    -   ``"manual"``: :class:`fiftyone.brain.visualization.ManualVisualizationConfig`

    You can pass ``create_index=True`` to create a spatial index of the
    computed points on your dataset's samples. This is highly recommended for
    large datasets as it enables efficient querying when lassoing points in
    embeddings plots. By default, spatial indexes are created in a field with
    name ``points_field=brain_key``, but you can customize this by manually
    providing a ``points_field``.

    You can also provide a ``points_field`` with ``create_index=False`` to
    store the points on your dataset without explicitly creating a database
    index. This will allow lasso callbacks to leverage point data rather than
    relying on ID selection, but without the added benefit of a database index
    to further optimize performance.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        patches_field (None): a sample field defining the image patches in each
            sample that have been/will be embedded. Must be of type
            :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines`
        embeddings (None): if no ``model`` is provided, this argument specifies
            pre-computed embeddings to use, which can be any of the following:

            -   a dict mapping sample IDs to embedding vectors
            -   a ``num_samples x num_embedding_dims`` array of embeddings
                corresponding to the samples in ``samples``
            -   if ``patches_field`` is specified, a dict mapping label IDs to
                to embedding vectors
            -   if ``patches_field`` is specified,  a dict mapping sample IDs
                to ``num_patches x num_embedding_dims`` arrays of patch
                embeddings
            -   the name of a dataset field containing the embeddings to use

            If a ``model`` is provided, this argument specifies the name of a
            field in which to store the computed embeddings. In either case,
            when working with patch embeddings, you can provide either the
            fully-qualified path to the patch embeddings or just the name of
            the label attribute in ``patches_field``
        points (None): a pre-computed low-dimensional representation to use. If
            provided, no embeddings will be used/computed. Can be any of the
            following:

            -   a dict mapping sample IDs to points vectors
            -   a ``num_samples x num_dims`` array of points corresponding to
                the samples in ``samples``
            -   if ``patches_field`` is specified, a dict mapping label IDs to
                points vectors
            -   if ``patches_field`` is specified, a ``num_patches x num_dims``
                array of points whose rows correspond to the flattened list of
                patches whose IDs are shown below::

                    # The list of patch IDs that the rows of `points` must match
                    _, id_field = samples._get_label_field_path(patches_field, "id")
                    patch_ids = samples.values(id_field, unwind=True)

        create_index (False): whether to create a spatial index for the
            computed points on your dataset
        points_field (None): an optional field name in which to store the
            spatial index. When ``create_index=True``, this defaults to
            ``points_field=brain_key``. When working with patches, you can
            provide either the fully-qualified path to the points field or just
            the name of the label attribute in ``patches_field``
        brain_key (None): a brain key under which to store the results of this
            method
        num_dims (2): the dimension of the visualization space
        method (None): the dimensionality reduction method to use. The
            supported values are
            ``fiftyone.brain.brain_config.visualization_methods.keys()`` and
            the default is
            ``fiftyone.brain.brain_config.default_visualization_method``
        similarity_index (None): a
            :class:`fiftyone.brain.similarity.SimilarityIndex` or the brain key
            of a similarity index to use to load pre-computed embeddings
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/index.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``patches_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``patches_field`` are specified
        batch_size (None): an optional batch size to use when computing
            embeddings. Only applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
        **kwargs: optional keyword arguments for the constructor of the
            :class:`fiftyone.brain.visualization.VisualizationConfig`
            being used

    Returns:
        a :class:`fiftyone.brain.visualization.VisualizationResults`
    """
    import fiftyone.brain.visualization as fbv

    return fbv.compute_visualization(
        samples,
        patches_field,
        embeddings,
        points,
        create_index,
        points_field,
        brain_key,
        num_dims,
        method,
        similarity_index,
        model,
        model_kwargs,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
        progress,
        **kwargs,
    )


def compute_similarity(
    samples,
    patches_field=None,
    roi_field=None,
    embeddings=None,
    brain_key=None,
    model=None,
    model_kwargs=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    progress=None,
    backend=None,
    **kwargs,
):
    """Uses embeddings to index the samples or their patches so that you can
    query/sort by similarity.

    Calling this method only creates the index. You can then call the methods
    exposed on the retuned :class:`fiftyone.brain.similarity.SimilarityIndex`
    object to perform the following operations:

    -   :meth:`sort_by_similarity() <fiftyone.brain.similarity.SimilarityIndex.sort_by_similarity>`:
        Sort the samples in the collection by similarity to a specific example
        or example(s)

    All indexes support querying by image similarity by passing sample IDs to
    :meth:`sort_by_similarity() <fiftyone.brain.similarity.SimilarityIndex.sort_by_similarity>`.
    In addition, if you pass the name of a model from the
    `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/index.html>`_
    like ``model="clip-vit-base32-torch"`` that can embed prompts to this
    method, then you can query the index by text similarity as well.

    In addition, if the backend supports it, you can call the following
    duplicate detection methods:

    -   :meth:`find_duplicates() <fiftyone.brain.similarity.DuplicatesMixin.find_duplicates>`:
        Query the index to find all examples with near-duplicates in the
        collection

    -   :meth:`find_unique() <fiftyone.brain.similarity.DuplicatesMixin.find_unique>`:
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
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute embeddings
        embeddings (None): embeddings to feed the index. This argument's
            behavior depends on whether a ``model`` is provided, as described
            below.

            If no ``model`` is provided, this argument specifies pre-computed
            embeddings to use:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``patches_field``/``roi_field`` is specified,  a dict
                mapping sample IDs to ``num_patches x num_dims`` arrays of
                patch embeddings
            -   the name of a dataset field from which to load embeddings
            -   ``None``: use the default model to compute embeddings
            -   ``False``: **do not** compute embeddings right now

            If a ``model`` is provided, this argument specifies where to store
            the model's embeddings:

            -   the name of a field in which to store the computed embeddings
            -   ``False``: **do not** compute embeddings right now

            In either case, when working with patch embeddings, you can provide
            either the fully-qualified path to the patch embeddings or just the
            name of the label attribute in ``patches_field``/``roi_field``
        brain_key (None): a brain key under which to store the results of this
            method
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/index.html>`_
            to use, or that was already used, to generate embeddings. The model
            must expose embeddings (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``patches_field``/``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``patches_field``/``roi_field`` are specified
        batch_size (None): an optional batch size to use when computing
            embeddings. Only applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
        backend (None): the similarity backend to use. The supported values are
            ``fiftyone.brain.brain_config.similarity_backends.keys()`` and the
            default is
            ``fiftyone.brain.brain_config.default_similarity_backend``
        **kwargs: keyword arguments for the
            :class:`fiftyone.brian.SimilarityConfig` subclass of the backend
            being used

    Returns:
        a :class:`fiftyone.brain.similarity.SimilarityIndex`
    """
    import fiftyone.brain.similarity as fbs

    return fbs.compute_similarity(
        samples,
        patches_field,
        roi_field,
        embeddings,
        brain_key,
        model,
        model_kwargs,
        force_square,
        alpha,
        batch_size,
        num_workers,
        skip_failures,
        progress,
        backend,
        **kwargs,
    )


def compute_near_duplicates(
    samples,
    threshold=0.2,
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
    """Detects potential duplicates in the given sample collection.

    Calling this method only initializes the index. You can then call the
    methods exposed on the returned object to perform the following operations:

    -   :meth:`duplicate_ids <fiftyone.brain.similarity.DuplicatesMixin.duplicate_ids>`:
        A list of duplicate IDs

    -   :meth:`neighbors_map <fiftyone.brain.similarity.DuplicatesMixin.neighbors_map>`:
        A dictionary mapping IDs to lists of ``(dup_id, dist)`` tuples

    -   :meth:`duplicates_view() <fiftyone.brain.similarity.DuplicatesMixin.duplicates_view>`:
        Returns a view of all duplicates in the input collection

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        threshold (0.2): the similarity distance threshold to use when
            detecting duplicates. Values in ``[0.1, 0.25]`` work well for the
            default setup
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute embeddings
        embeddings (None): if no ``model`` is provided, this argument specifies
            pre-computed embeddings to use, which can be any of the following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``roi_field`` is specified,  a dict mapping sample IDs to
                ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

            If a ``model`` is provided, this argument specifies the name of a
            field in which to store the computed embeddings. In either case,
            when working with patch embeddings, you can provide either the
            fully-qualified path to the patch embeddings or just the name of
            the label attribute in ``roi_field``
        similarity_index (None): a
            :class:`fiftyone.brain.similarity.SimilarityIndex` or the brain key
            of a similarity index to use to load pre-computed embeddings
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/models.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``roi_field`` are specified
        batch_size (None): a batch size to use when computing embeddings. Only
            applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead

    Returns:
        a :class:`fiftyone.brain.similarity.SimilarityIndex`
    """
    import fiftyone.brain.internal.core.duplicates as fbd

    return fbd.compute_near_duplicates(
        samples,
        threshold=threshold,
        roi_field=roi_field,
        embeddings=embeddings,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
        force_square=force_square,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=skip_failures,
        progress=progress,
    )


def compute_exact_duplicates(
    samples,
    num_workers=None,
    skip_failures=True,
    progress=None,
):
    """Detects duplicate media in a sample collection.

    This method detects exact duplicates with the same filehash. Use
    :meth:`compute_near_duplicates` to detect near-duplicates.

    If duplicates are found, the first instance in ``samples`` will be the key
    in the returned dictionary, while the subsequent duplicates will be the
    values in the corresponding list.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        num_workers (None): an optional number of processes to use
        skip_failures (True): whether to gracefully ignore samples whose
            filehash cannot be computed
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead

    Returns:
        a dictionary mapping IDs of samples with exact duplicates to lists of
        IDs of the duplicates for the corresponding sample
    """
    import fiftyone.brain.internal.core.duplicates as fbd

    return fbd.compute_exact_duplicates(
        samples, num_workers, skip_failures, progress
    )


def compute_leaky_splits(
    samples,
    splits,
    threshold=0.2,
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
    """Computes potential leaks between splits of the given sample collection.

    Calling this method only initializes the index. You can then call the
    methods exposed on the returned object to perform the following operations:

    -   :meth:`leaks_view() <fiftyone.brain.core.internal.leaky_splits.LeakySplitsIndex.leaks_view>`:
        Returns a view of all leaks in the input collection

    -   :meth:`no_leaks_view() <fiftyone.brain.core.internal.leaky_splits.LeakySplitsIndex.no_leaks_view>`:
        Returns the subset of the input collection without any leaks

    -   :meth:`leaks_for_sample() <fiftyone.brain.core.internal.leaky_splits.LeakySplitsIndex.leaks_for_sample>`:
        Returns a view with leaks corresponding to the given sample

    -   :meth:`tag_leaks() <fiftyone.brain.core.internal.leaky_splits.LeakySplitsIndex.tag_leaks>`:
        Tags leaks in the dataset as leaks

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        splits: the dataset splits, specified in one of the following ways:

            -   a list of tag strings
            -   the name of a string/list field that encodes the split
                memberships
            -   a dict mapping split names to
                :class:`fiftyone.core.view.DatasetView` instances
        threshold (0.2): the similarity distance threshold to use when
            detecting leaks. Values in ``[0.1, 0.25]`` work well for the
            default setup
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute leaks
        embeddings (None): if no ``model`` is provided, this argument specifies
            pre-computed embeddings to use, which can be any of the following:

            -   a ``num_samples x num_dims`` array of embeddings
            -   if ``roi_field`` is specified,  a dict mapping sample IDs to
                ``num_patches x num_dims`` arrays of patch embeddings
            -   the name of a dataset field containing the embeddings to use

            If a ``model`` is provided, this argument specifies the name of a
            field in which to store the computed embeddings. In either case,
            when working with patch embeddings, you can provide either the
            fully-qualified path to the patch embeddings or just the name of
            the label attribute in ``roi_field``
        similarity_index (None): a
            :class:`fiftyone.brain.similarity.SimilarityIndex` or the brain key
            of a similarity index to use to load pre-computed embeddings
        model (None): a :class:`fiftyone.core.models.Model` or the name of a
            model from the
            `FiftyOne Model Zoo <https://docs.voxel51.com/user_guide/model_zoo/models.html>`_
            to use to generate embeddings. The model must expose embeddings
            (``model.has_embeddings = True``)
        model_kwargs (None): a dictionary of optional keyword arguments to pass
            to the model's ``Config`` when a model name is provided
        force_square (False): whether to minimally manipulate the patch
            bounding boxes into squares prior to extraction. Only applicable
            when a ``model`` and ``roi_field`` are specified
        alpha (None): an optional expansion/contraction to apply to the patches
            before extracting them, in ``[-1, inf)``. If provided, the length
            and width of the box are expanded (or contracted, when
            ``alpha < 0``) by ``(100 * alpha)%``. For example, set
            ``alpha = 0.1`` to expand the boxes by 10%, and set
            ``alpha = -0.1`` to contract the boxes by 10%. Only applicable when
            a ``model`` and ``roi_field`` are specified
        batch_size (None): a batch size to use when computing embeddings. Only
            applicable when a ``model`` is provided
        num_workers (None): the number of workers to use when loading images.
            Only applicable when a Torch-based model is being used to compute
            embeddings
        skip_failures (True): whether to gracefully continue without raising an
            error if embeddings cannot be generated for a sample
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead

    Returns:
        a :class:`fiftyone.brain.internal.core.leaky_splits.LeakySplitsIndex`
    """
    import fiftyone.brain.internal.core.leaky_splits as fbl

    return fbl.compute_leaky_splits(
        samples,
        splits,
        threshold=threshold,
        roi_field=roi_field,
        embeddings=embeddings,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
        force_square=force_square,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=skip_failures,
        progress=progress,
    )
