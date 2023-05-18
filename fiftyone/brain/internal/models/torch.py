"""
PyTorch utilities.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone.utils.torch as fout

from fiftyone.brain.internal.models import HasBrainModel

import torch


class TorchImageModelConfig(fout.TorchImageModelConfig, HasBrainModel):
    """Configuration for running a :class:`TorchImageModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    parameters.

    Args:
        model_name (None): the name of the Brain model state dict to load
        model_path (None): the path to a state dict on disk to load
    """

    def __init__(self, d):
        d = self.init(d)
        super().__init__(d)


class TorchImageModel(fout.TorchImageModel):
    """Wrapper for evaluating a Torch model on images whose state dict is
    stored privately by the Brain.

    Args:
        config: an :class:`TorchImageModelConfig`
    """

    def _download_model(self, config):
        config.download_model_if_necessary()

    def _load_state_dict(self, model, config):
        state_dict = torch.load(config.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
