"""
Indexing a video collection by a temporal label field.

A temporal label addresses a span of a video without reference to a frame, so a
video collection can be indexed by one. Every other patches field describes a
region within a frame and requires a frames view. These tests pin both halves,
and the round trip such an index is built for.

Samples are synthetic and no media is read. Only
``test_the_index_records_the_model`` names a zoo model, and loading one is the
only reason this file touches the model zoo at all.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.media as fom
import fiftyone.zoo as foz
import fiftyone.brain.internal.core.utils as fbu

SEGMENTS_FIELD = "auto_embedding_segments"
SEGMENTS_PER_SAMPLE = 2
EMBEDDING_DIM = 8
MODEL = "clip-vit-base32-torch"


def _temporal_labels():
    return fo.TemporalDetections(
        detections=[
            fo.TemporalDetection(support=[1 + 30 * i, 30 + 30 * i])
            for i in range(SEGMENTS_PER_SAMPLE)
        ]
    )


def _region_labels():
    return fo.Detections(
        detections=[
            fo.Detection(bounding_box=[0.1, 0.1, 0.2, 0.2])
            for _ in range(SEGMENTS_PER_SAMPLE)
        ]
    )


def _dataset(filepath, make_labels, num_samples=4):
    dataset = fo.Dataset()
    for i in range(num_samples):
        sample = fo.Sample(filepath=filepath % i)
        sample[SEGMENTS_FIELD] = make_labels()
        dataset.add_sample(sample)

    return dataset


@pytest.fixture
def video_segments():
    """A video dataset whose segment field holds temporal labels."""
    dataset = _dataset("/tmp/fob_vid%d.mp4", _temporal_labels)
    try:
        yield dataset
    finally:
        dataset.delete()


@pytest.fixture
def video_regions():
    """A video dataset whose field holds frame regions instead of segments."""
    dataset = _dataset("/tmp/fob_vid%d.mp4", _region_labels)
    try:
        yield dataset
    finally:
        dataset.delete()


@pytest.fixture
def image_regions():
    """An image dataset, which the temporal rule must leave alone."""
    dataset = _dataset("/tmp/fob_img%d.png", _region_labels)
    try:
        yield dataset
    finally:
        dataset.delete()


def _populated_index(dataset):
    results = fob.compute_similarity(
        dataset,
        patches_field=SEGMENTS_FIELD,
        embeddings=False,
        brain_key="segments",
    )

    label_ids = dataset.values(
        "%s.detections.id" % SEGMENTS_FIELD, unwind=True
    )
    sample_ids = [
        sample_id
        for sample_id in dataset.values("id")
        for _ in range(SEGMENTS_PER_SAMPLE)
    ]
    embeddings = np.random.rand(len(label_ids), EMBEDDING_DIM).astype(
        "float32"
    )
    results.add_to_index(embeddings, sample_ids, label_ids=label_ids)

    return results, label_ids


class TestWhichFieldsAVideoCollectionAccepts:
    def test_a_singular_temporal_label_is_accepted(self):
        dataset = _dataset(
            "/tmp/fob_vid%d.mp4",
            lambda: fo.TemporalDetection(support=[1, 30]),
        )
        try:
            fbu._validate_patches_args(dataset, SEGMENTS_FIELD)
        finally:
            dataset.delete()

    def test_temporal_labels_are_accepted(self, video_segments):
        assert video_segments.media_type == fom.VIDEO
        fbu._validate_patches_args(video_segments, SEGMENTS_FIELD)

    def test_frame_regions_are_refused(self, video_regions):
        with pytest.raises(ValueError, match="sample-level temporal label"):
            fbu._validate_patches_args(video_regions, SEGMENTS_FIELD)

    def test_the_refusal_names_the_field_and_the_type_found(
        self, video_regions
    ):
        with pytest.raises(ValueError) as excinfo:
            fbu._validate_patches_args(video_regions, SEGMENTS_FIELD)

        message = str(excinfo.value)
        assert SEGMENTS_FIELD in message
        assert "Detections" in message
        # the caller is told what to do, not only what went wrong
        assert "to_frames()" in message

    @pytest.mark.parametrize(
        "label_cls", [fo.Detections, fo.TemporalDetections]
    )
    def test_a_frame_level_field_is_refused(self, video_segments, label_cls):
        # a frame field addresses one frame, so it is not a segment even when
        # its own type is temporal
        video_segments.add_frame_field(
            "objects",
            fo.EmbeddedDocumentField,
            embedded_doc_type=label_cls,
        )
        with pytest.raises(ValueError, match="sample-level temporal label"):
            fbu._validate_patches_args(video_segments, "frames.objects")

    def test_image_collections_apply_the_region_rule(self, image_regions):
        assert image_regions.media_type == fom.IMAGE
        fbu._validate_patches_args(image_regions, SEGMENTS_FIELD)


class TestComputingEmbeddingsIsRefused:
    """Cropping a frame region is the only patch embedding that exists, so a
    caller must supply embeddings for temporal labels rather than ask for them.
    """

    def test_a_model_without_embeddings_is_refused(self, video_segments):
        with pytest.raises(ValueError, match="Supply `embeddings`"):
            fob.compute_similarity(
                video_segments,
                patches_field=SEGMENTS_FIELD,
                model=MODEL,
                brain_key="segments",
            )


class TestViewsThatCropLabels:
    """`to_patches` crops each label, which a temporal label does not
    describe, so the duplicate and unique views have nothing to return.
    """

    # each finder clears the other's results, so they cannot share a setup
    @pytest.mark.parametrize(
        "find,view_method",
        [
            (
                lambda results: results.find_duplicates(thresh=0.1),
                "duplicates_view",
            ),
            (lambda results: results.find_unique(1), "unique_view"),
        ],
        ids=["duplicates", "unique"],
    )
    def test_they_refuse_a_video_index(
        self, video_segments, find, view_method
    ):
        results, _ = _populated_index(video_segments)
        find(results)

        with pytest.raises(ValueError, match="no patches view"):
            getattr(results, view_method)()


class TestMediaTypesThatAreNotVideo:
    """The temporal rule is scoped to video; nothing else changes shape."""

    def test_a_multimodal_collection_is_unchanged(self):
        dataset = _dataset("/tmp/fob_scene%d.mcap", _region_labels)
        try:
            assert dataset.media_type == fom.MULTIMODAL
            fbu._validate_patches_args(dataset, SEGMENTS_FIELD)
        finally:
            dataset.delete()


class TestFieldsThatDoNotResolve:
    """Consulting the label type puts two new failures on the video branch."""

    def test_an_unknown_field_says_so(self, video_segments):
        with pytest.raises(ValueError, match="has no field"):
            fbu._validate_patches_args(video_segments, "not_a_field")

    def test_a_non_label_field_says_so(self, video_segments):
        with pytest.raises(ValueError, match="not a Label type"):
            fbu._validate_patches_args(video_segments, "filepath")


class TestSegmentIndexRoundTrip:
    def test_the_index_records_the_model(self, video_segments):
        # the zoo load is stubbed: the index records the name, and loading the
        # weights to probe prompt support is not what this pins
        with mock.patch.object(foz, "load_zoo_model", return_value=None):
            results = fob.compute_similarity(
                video_segments,
                patches_field=SEGMENTS_FIELD,
                embeddings=False,
                model=MODEL,
                brain_key="segments",
            )

        assert results.config.model == MODEL
        assert results.total_index_size == 0

    def test_one_row_is_added_per_segment(self, video_segments):
        results, label_ids = _populated_index(video_segments)

        assert len(label_ids) == len(video_segments) * SEGMENTS_PER_SAMPLE
        assert results.total_index_size == len(label_ids)

    def test_a_view_scoped_index_resolves_its_ids(self, video_segments):
        # querying validates too, through _validate_args on each get_ids call,
        # so a view-scoped index is the half of the fix a build never reaches
        results, _ = _populated_index(video_segments)

        results.use_view(video_segments.limit(2))

        # one entry per label, so the parent id repeats once per segment
        assert len(results.current_sample_ids) == 2 * SEGMENTS_PER_SAMPLE
        assert len(set(results.current_sample_ids)) == 2
        assert len(results.current_label_ids) == 2 * SEGMENTS_PER_SAMPLE

    def test_a_query_selects_parents_and_filters_their_labels(
        self, video_segments
    ):
        results, label_ids = _populated_index(video_segments)

        view = results.sort_by_similarity(label_ids[0], k=4)

        stages = [stage.__class__.__name__ for stage in view._stages]
        assert stages == ["Select", "FilterLabels"]
        # k counts labels, so the parents carrying them are fewer
        assert view.count("%s.detections" % SEGMENTS_FIELD) == 4
        assert len(view) < len(video_segments) * SEGMENTS_PER_SAMPLE
