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
        keep_ids: an array of sample/patch IDs indicating the non-duplicates
        thresh: the embedding distance threshold used
        config: the :class:`DuplicatesConfig` used
        neighbors (None): the ``sklearn.neighbors.NearestNeighbors`` instance
    """

    def __init__(
        self, samples, embeddings, keep_ids, thresh, config, neighbors=None,
    ):
        self._samples = samples
        self._neighbors = neighbors

        self.embeddings = embeddings
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
        keep_ids = np.array(d["keep_ids"])
        thresh = d["thresh"]
        config = fbd.DuplicatesConfig.from_dict(d["config"])
        return cls(samples, embeddings, keep_ids, thresh, config)


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
