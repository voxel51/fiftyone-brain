"""
Point selection utilities.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import math

import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import sklearn.metrics.pairwise as skp

from fiftyone import ViewField as F
from fiftyone.core.expressions import ObjectId


class PointSelector(object):
    """Class that serves an interactive UI for selecting points in a matplotlib
    plot.

    You can provide a ``session`` object together with one of the following to
    link the currently selected points to a FiftyOne App instance:

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
        expand_selected (2.0): expand the size of selected points by this
            amount
        click_tolerance (0.02): a click distance tolerance in ``[0, 1]`` when
            clicking individual points
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
        expand_selected=2.0,
        click_tolerance=0.02,
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
        self.expand_selected = expand_selected
        self.click_tolerance = click_tolerance

        self._canvas = ax.figure.canvas
        self._xy = collection.get_offsets()
        self._num_pts = len(self._xy)
        self._fc = collection.get_facecolors()
        self._ms = collection.get_sizes()
        self._init_ms = self._ms[0]
        self._click_thresh = click_tolerance * min(
            np.max(self._xy, axis=0) - np.min(self._xy, axis=0)
        )

        self._inds = np.array([], dtype=int)
        self._selected_sample_ids = None
        self._selected_object_ids = None
        self._canvas.mpl_connect("close_event", lambda e: self._disconnect())

        self._connected = False
        self._session = None
        self._init_view = None
        self._lasso = None
        self._shift = False
        self._keypress_events = []
        self.connect()

    @property
    def is_connected(self):
        """Whether this selector is currently linked to its plot and session
        (if any).
        """
        return self._connected

    @property
    def has_linked_session(self):
        """Whether this object has a linked
        :class:`fiftone.core.session.Session` that can be updated when points
        are selected.
        """
        return self._session is not None

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
    def any_selected(self):
        """Whether any points are currently selected."""
        return self._inds.size > 0

    @property
    def selected_inds(self):
        """A list of indices of the currently selected points."""
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

    def select_samples(self, sample_ids):
        """Selects the points corresponding to the given sample IDs.

        Args:
            sample_ids: a list of sample IDs
        """
        if not self.is_selecting_samples:
            raise ValueError("This selector cannot select samples")

        x = np.expand_dims(self.sample_ids, axis=1)
        y = np.expand_dims(sample_ids, axis=0)
        inds = np.nonzero(np.any(x == y, axis=1))[0]
        self._select_inds(inds)

    def select_objects(self, object_ids):
        """Selects the points corresponding to the objects with the given IDs.

        Args:
            object_ids: a list of object IDs
        """
        if not self.is_selecting_objects:
            raise ValueError("This selector cannot select objects")

        x = np.expand_dims(self.object_ids, axis=1)
        y = np.expand_dims(object_ids, axis=0)
        inds = np.nonzero(np.any(x == y, axis=1))[0]
        self._select_inds(inds)

    def tag_selected(self, tag):
        """Adds the tag to the currently selected samples/objects.

        Args:
            tag: a tag
        """
        view = self.selected_view()

        if view is None:
            return

        if self.is_selecting_samples:
            view.tag_samples(tag)

        if self.is_selecting_objects:
            view.tag_objects(self.object_field, tag)

        self.refresh()

    def selected_view(self):
        """Returns a :class:`fiftyone.core.view.DatasetView` containing the
        currently selected samples/objects.

        Returns:
            a :class:`fiftyone.core.view.DatasetView`, or None if no points are
            selected
        """
        if not self.has_linked_session:
            raise ValueError("This selector is not linked to a session")

        if not self.any_selected:
            return None

        if self.is_selecting_samples:
            return self._init_view.select(self._selected_sample_ids)

        if self.is_selecting_objects:
            _object_ids = [ObjectId(_id) for _id in self._selected_object_ids]
            return self._init_view.select_fields(
                self.object_field
            ).filter_labels(self.object_field, F("_id").is_in(_object_ids))

        return None

    def connect(self):
        """Connects this selector to its plot and session (if any)."""
        if self.is_connected:
            return

        session = self.session
        if session is not None:
            if session.view is not None:
                self._init_view = session.view
            else:
                self._init_view = session.dataset.view()

        self._lasso = LassoSelector(self.ax, onselect=self._onselect)
        self._session = session
        self._keypress_events = [
            self._canvas.mpl_connect("key_press_event", self._onkeypress),
            self._canvas.mpl_connect("key_release_event", self._onkeyrelease),
        ]
        self._connected = True

        self.ax.set_title("Click or drag to select points")
        self._canvas.draw_idle()

    def refresh(self):
        """Refreshes the selector's plot and linked session (if any)."""
        self._canvas.draw_idle()
        self._update_session()

    def disconnect(self,):
        """Disconnects this selector from its plot and sesssion (if any)."""
        if not self.is_connected:
            return

        self._lasso.disconnect_events()
        self._lasso = None

        for cid in self._keypress_events:
            self._canvas.mpl_disconnect(cid)

        self._shift = False
        self._keypress_events = []
        self.ax.set_title("")

        self._fc[:, -1] = 1
        self.collection.set_facecolors(self._fc)
        self._canvas.draw_idle()

        self._disconnect()

    def _disconnect(self):
        self._session = None
        self._connected = False

    def _onkeypress(self, event):
        if event.key == "shift":
            self._shift = True
            self.ax.set_title("Click or drag to add/remove points")
            self._canvas.draw_idle()

    def _onkeyrelease(self, event):
        if event.key == "shift":
            self._shift = False
            self.ax.set_title("Click or drag to select points")
            self._canvas.draw_idle()

    def _onselect(self, vertices):
        if self._is_click(vertices):
            dists = skp.euclidean_distances(self._xy, np.array([vertices[0]]))
            click_ind = np.argmin(dists)
            if dists[click_ind] < self._click_thresh:
                inds = [click_ind]
            else:
                inds = []

            inds = np.array(inds, dtype=int)
        else:
            path = Path(vertices)
            inds = np.nonzero(path.contains_points(self._xy))[0]

        self._select_inds(inds)

    @staticmethod
    def _is_click(vertices):
        if len(vertices) > 2:
            return False

        return math.isclose(vertices[0][0], vertices[-1][0]) and math.isclose(
            vertices[0][1], vertices[-1][1]
        )

    def _select_inds(self, inds):
        if self._shift:
            new_inds = set(inds)
            inds = set(self._inds)
            if new_inds.issubset(inds):
                # the new selection is a subset of the current selection, so
                # remove the selection
                inds.difference_update(new_inds)
            else:
                # the new selection contains new points, so add them
                inds.update(new_inds)

            inds = np.array(sorted(inds), dtype=int)
        else:
            inds = np.unique(inds)

        if np.all(inds == self._inds):
            self._canvas.draw_idle()
            return

        self._inds = inds
        self._update_selections()

        self._prep_collection()

        if inds.size == 0:
            self._fc[:, -1] = 1
        else:
            self._fc[:, -1] = self.alpha_other
            self._fc[inds, -1] = 1

        self.collection.set_facecolors(self._fc)

        if self.expand_selected is not None:
            self._ms[:] = self._init_ms
            self._ms[inds] = self.expand_selected * self._init_ms

        self.collection.set_sizes(self._ms)

        self.refresh()

    def _update_selections(self):
        if self.is_selecting_samples:
            self._selected_sample_ids = list(self.sample_ids[self._inds])

        if self.is_selecting_objects:
            self._selected_object_ids = list(self.object_ids[self._inds])

    def _update_session(self):
        if not self.has_linked_session:
            return

        if self.any_selected:
            view = self.selected_view()
        else:
            view = self._init_view

        self._session.view = view

    def _prep_collection(self):
        # @todo why is this necessary? We do this JIT here because it seems
        # that when __init__() runs, `get_facecolors()` doesn't have all the
        # data yet...
        if len(self._fc) < self._num_pts:
            self._fc = self.collection.get_facecolors()

        if len(self._fc) < self._num_pts:
            self._fc = np.tile(self._fc[0], (self._num_pts, 1))

        if self.expand_selected is not None:
            if len(self._ms) < self._num_pts:
                self._ms = np.tile(self._ms[0], self._num_pts)
