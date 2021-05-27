"""
Duplicates interface.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.utils as etau

import fiftyone.core.brain as fob


class DuplicatesResults(fob.BrainResults):
    """Class storing the results of :meth:`fiftyone.brain.compute_duplicates`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        embeddings: a ``num_embeddings x num_dims`` array of embeddings
        config: the :class:`DuplicatesConfig` used
        dup_ids (None): an array of duplicate sample/patch IDs
        keep_ids (None): an array of non-duplicate sample/patch IDs
        thresh (None): the embedding distance threshold used
        neighbors (None): a ``sklearn.neighbors.NearestNeighbors`` instance
    """

    def __init__(
        self,
        samples,
        embeddings,
        config,
        dup_ids=None,
        keep_ids=None,
        thresh=None,
        neighbors=None,
    ):
        self._samples = samples
        self._neighbors = neighbors

        self.embeddings = embeddings
        self.config = config
        self.dup_ids = dup_ids
        self.keep_ids = keep_ids
        self.thresh = thresh
        self.config = config

    def _init_neighbors(self):
        import fiftyone.brain.internal.core.duplicates as fbd

        self._neighbors = fbd.init_neighbors(
            self.embeddings, self.config.metric
        )

    @classmethod
    def _from_dict(cls, d, samples):
        import fiftyone.brain.internal.core.duplicates as fbd

        embeddings = np.array(d["embeddings"])
        config = fbd.DuplicatesConfig.from_dict(d["config"])

        dup_ids = d.get("dup_ids", None)
        if dup_ids is not None:
            dup_ids = np.array(dup_ids)

        keep_ids = d.get("keep_ids", None)
        if keep_ids is not None:
            keep_ids = np.array(keep_ids)

        thresh = d.get("thresh", None)

        return cls(
            samples,
            embeddings,
            config,
            dup_ids=dup_ids,
            keep_ids=keep_ids,
            thresh=thresh,
        )


class DuplicatesConfig(fob.BrainMethodConfig):
    """Duplicates configuration.

    Args:
        metric (None): the embedding distance metric used
        thresh (None): the distance threshold to use to determine duplicates
        fraction (None): the desired fraction of images/patches to tag as
            duplicates
        embeddings_field (None): the sample field containing the embeddings
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings
        patches_field (None): the sample field defining the patches we're
            assessing for duplicates
    """

    def __init__(
        self,
        metric=None,
        thresh=None,
        fraction=None,
        embeddings_field=None,
        model=None,
        patches_field=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        super().__init__(**kwargs)
        self.metric = metric
        self.thresh = thresh
        self.fraction = fraction
        self.embeddings_field = embeddings_field
        self.patches_field = patches_field

    @property
    def method(self):
        return "duplicates"

    @property
    def run_cls(self):
        from fiftyone.brain.internal.core.duplicates import Duplicates

        return Duplicates
