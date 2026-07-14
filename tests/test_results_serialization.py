"""
Tests for :class:`VisualizationResults` serialization.

Array-valued fields (points and ids) serialize as compressed ``.npy``
strings; results written by earlier versions stored them as JSON lists.
Both encodings must load, and the round trip must be lossless.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import unittest

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


def _make_dataset():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=20).clone()
    dataset.persistent = False
    return dataset


def _compute(dataset, brain_key, dims=2):
    points = np.random.RandomState(51).rand(len(dataset), dims)
    return fob.compute_visualization(
        dataset, points=points, brain_key=brain_key, verbose=False
    )


class ResultsSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()

    @classmethod
    def tearDownClass(cls):
        cls.dataset.delete()

    def test_round_trip_through_the_database(self):
        results = _compute(self.dataset, "viz_roundtrip")

        loaded = self.dataset.load_brain_results(
            "viz_roundtrip", cache=False
        )

        np.testing.assert_array_equal(loaded.points, results.points)
        np.testing.assert_array_equal(loaded.sample_ids, results.sample_ids)
        self.assertIsNone(loaded.label_ids)

    def test_arrays_serialize_as_strings_not_lists(self):
        results = _compute(self.dataset, "viz_encoding")

        d = results.serialize()

        self.assertIsInstance(d["points"], str)
        self.assertIsInstance(d["sample_ids"], str)
        self.assertIsNone(d["label_ids"])

    def test_legacy_list_encoding_still_loads(self):
        results = _compute(self.dataset, "viz_legacy")

        # Results written by earlier versions stored the arrays as JSON
        # lists; loading them must remain equivalent forever
        d = results.serialize()
        d["points"] = results.points.tolist()
        d["sample_ids"] = np.asarray(results.sample_ids).tolist()
        d["label_ids"] = None

        loaded = type(results)._from_dict(
            d, self.dataset, results.config, "viz_legacy"
        )

        np.testing.assert_array_equal(loaded.points, results.points)
        np.testing.assert_array_equal(loaded.sample_ids, results.sample_ids)
        self.assertIsNone(loaded.label_ids)

    def test_3d_points_round_trip(self):
        results = _compute(self.dataset, "viz_3d_roundtrip", dims=3)

        loaded = self.dataset.load_brain_results(
            "viz_3d_roundtrip", cache=False
        )

        self.assertEqual(loaded.points.shape, (len(self.dataset), 3))
        np.testing.assert_array_equal(loaded.points, results.points)


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)
