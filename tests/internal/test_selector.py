"""
Tests for the :mod:`fiftyone.brain.internal.core.selector` module.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import matplotlib.pyplot as plt
import numpy as np

import fiftyone as fo

from fiftyone.brain.internal.core.plot.selector import PointSelector


def test_point_selector():
    fig, ax = plt.subplots()

    data = np.random.rand(100, 2)
    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = PointSelector(ax, pts)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.selected_inds)
            selector.disconnect()

    fig.canvas.mpl_connect("key_press_event", accept)

    plt.show(block=False)

    input("Press enter to continue...")


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
