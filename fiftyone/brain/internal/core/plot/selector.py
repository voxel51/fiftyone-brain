"""
Point selection utilities.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from fiftyone import ViewField as F
from fiftyone.core.expressions import ObjectId


class PointSelector(object):
    """Class that serves an interactive UI for selecting points in a matplotlib
    plot.

    You can provide a ``session`` object and one of the following to link the
    currently selected points to a FiftyOne App instance:

    -   Sample selection: If ``sample_ids`` is provided, then when points are
        selected, a view containing the corresponding samples will be loaded in
        the App
    -   Object selection: If ``object_ids`` and ``object_field`` are provided,
        then when points are selected, a view containing the corresponding
        objects in ``object_field`` will be loaded in the App

    Args:
        ax: a matplotlib axis
        collection: a ``matplotlib.collections.Collection`` to select points
            from
        session (None): a :class:`fiftyone.core.session.Session` to link with
            this selector
        sample_ids (None): an array of sample IDs corresponding to
            ``collection``
        object_ids (None): an array of object IDs corresponding to
            ``collection``
        object_field (None): the sample field containing the objects in
            ``collection``
        alpha_other (0.25): a transparency value for unselected points
    """

    def __init__(
        self,
        ax,
        collection,
        session=None,
        sample_ids=None,
        object_ids=None,
        object_field=None,
        alpha_other=0.25,
    ):
        if sample_ids is not None:
            sample_ids = np.asarray(sample_ids)

        if object_ids is not None:
            object_ids = np.asarray(object_ids)

        self.ax = ax
        self.collection = collection
        self.session = session
        self.sample_ids = sample_ids
        self.object_ids = object_ids
        self.object_field = object_field
        self.alpha_other = alpha_other

        self._init_view = None
        if session is not None:
            if session.view is not None:
                self._init_view = session.view
            else:
                self._init_view = session.dataset.view()

        self._canvas = ax.figure.canvas
        self._xy = collection.get_offsets()
        self._num_pts = len(self._xy)
        self._fc = collection.get_facecolors()

        self._lasso = LassoSelector(ax, onselect=self._onselect)
        self._inds = np.array([], dtype=int)
        self._selected_sample_ids = None
        self._selected_object_ids = None

    @property
    def has_linked_session(self):
        """Whether this object has a linked
        :class:`fiftone.core.session.Session` that can be updated when points
        are selected.
        """
        return self.session is not None

    @property
    def is_selecting_samples(self):
        """Whether this selector is selecting samples from a collection."""
        return self.sample_ids is not None

    @property
    def is_selecting_objects(self):
        """Whether this selector is selecting objects from a field of a sample
        collection.
        """
        return self.object_ids is not None and self.object_field is not None

    @property
    def selected_inds(self):
        """An array containing the indices of the currently selected points."""
        return list(self._inds)

    @property
    def selected_samples(self):
        """A list of the currently selected samples, or None if
        :meth:`is_selecting_samples` is False.
        """
        return self._selected_sample_ids

    @property
    def selected_objects(self):
        """A list of the currently selected objects, or None if
        :meth:`is_selecting_objects` is False.
        """
        return self._selected_object_ids

    def disconnect(self):
        """Disconnects this selector from its plot and linked sesssion (if any).
        """
        self._lasso.disconnect_events()
        self._fc[:, -1] = 1
        self.collection.set_facecolors(self._fc)
        self._canvas.draw_idle()
        self._reset_session()
        self.session = None

    def _onselect(self, vertices):
        path = Path(vertices)
        self._inds = np.nonzero(path.contains_points(self._xy))[0]
        self._check_facecolors()
        if self._inds.size == 0:
            self._fc[:, -1] = 1
        else:
            self._fc[:, -1] = self.alpha_other
            self._fc[self._inds, -1] = 1

        self.collection.set_facecolors(self._fc)
        self._canvas.draw_idle()
        self._update_session()

    def _update_session(self):
        if not self.has_linked_session:
            return

        if self._inds.size == 0:
            self._reset_session()
            return

        if self.is_selecting_samples:
            self._selected_sample_ids = list(self.sample_ids[self._inds])
            self.session.view = self._init_view.select(
                self._selected_sample_ids
            )

        if self.is_selecting_objects:
            self._selected_object_ids = list(self.object_ids[self._inds])
            _object_ids = [ObjectId(_id) for _id in self._selected_object_ids]
            self.session.view = self._init_view.filter_labels(
                self.object_field, F("_id").is_in(_object_ids)
            )

    def _reset_session(self):
        if not self.has_linked_session:
            return

        self.session.view = self._init_view

    def _check_facecolors(self):
        # @todo why is this necessary? We do this JIT here because it seems
        # that when __init__() runs, `get_facecolors()` doesn't have all the
        # data yet...
        if len(self._fc) < self._num_pts:
            self._fc = self.collection.get_facecolors()

        if len(self._fc) < self._num_pts:
            self._fc = np.tile(self._fc, (self._num_pts, 1))
