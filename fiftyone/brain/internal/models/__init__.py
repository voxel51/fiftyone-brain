"""
Brain models.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import deepcopy
import logging
import os

from eta.core.config import ConfigError
import eta.core.learning as etal
import eta.core.models as etam

import fiftyone.core.models as fom


logger = logging.getLogger(__name__)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_MANIFEST_PATH = os.path.join(_THIS_DIR, "manifest.json")
_MODELS_DIR = os.path.join(_THIS_DIR, "cache")


def list_models():
    """Returns the list of available models.

    Returns:
        a list of model names
    """
    manifest = _load_models_manifest()
    return sorted([model.name for model in manifest])


def list_downloaded_models():
    """Returns information about the models that have been downloaded.

    Returns:
        a dict mapping model names to (model path, ``eta.core.models.Model``)
        tuples
    """
    manifest = _load_models_manifest()
    models = {}
    for model in manifest:
        if model.is_in_dir(_MODELS_DIR):
            model_path = model.get_path_in_dir(_MODELS_DIR)
            models[model.name] = (model_path, model)

    return models


def is_model_downloaded(name):
    """Determines whether the model of the given name is downloaded.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used

    Returns:
        True/False
    """
    model = _get_model(name)
    return model.is_in_dir(_MODELS_DIR)


def download_model(name, overwrite=False):
    """Downloads the model of the given name.

    If the model is already downloaded, it is not re-downloaded unless
    ``overwrite == True`` is specified.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used. Call
            :func:`list_models` to see the available models
        overwrite (False): whether to overwrite any existing files

    Returns:
        tuple of

        -   model: the ``eta.core.models.Model`` instance for the model
        -   model_path: the path to the downloaded model on disk
    """
    model, model_path = _get_model_in_dir(name)

    if not overwrite and is_model_downloaded(name):
        logger.info("Model '%s' is already downloaded", name)
    else:
        model.manager.download_model(model_path, force=overwrite)

    return model, model_path


def install_model_requirements(name, error_level=0):
    """Installs any package requirements for the model with the given name.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used. Call
            :func:`list_models` to see the available models
        error_level: the error level to use, defined as:

            0: raise error if a requirement install fails
            1: log warning if a requirement install fails
            2: ignore install fails
    """
    model = _get_model(name)
    model.install_requirements(error_level=error_level)


def ensure_model_requirements(name, error_level=0):
    """Ensures that the package requirements for the model with the given name
    are satisfied.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used. Call
            :func:`list_models` to see the available models
        error_level: the error level to use, defined as:

            0: raise error if a requirement is not satisfied
            1: log warning if a requirement is not satisifed
            2: ignore unsatisifed requirements
    """
    model = _get_model(name)
    model.ensure_requirements(error_level=error_level)


def load_model(
    name,
    download_if_necessary=True,
    install_requirements=False,
    error_level=0,
    **kwargs
):
    """Loads the model of the given name.

    By default, the model will be downloaded if necessary.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used. Call
            :func:`list_models` to see the available models
        download_if_necessary (True): whether to download the model if it is
            not found in the specified directory
        install_requirements: whether to install any requirements before
            loading the model. By default, this is False
        error_level: the error level to use, defined as:

            0: raise error if a requirement is not satisfied
            1: log warning if a requirement is not satisifed
            2: ignore unsatisifed requirements

        **kwargs: keyword arguments to inject into the model's ``Config``
            instance

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    model = _get_model(name)

    if not model.is_in_dir(_MODELS_DIR):
        if not download_if_necessary:
            raise ValueError("Model '%s' is not downloaded" % name)

        download_model(name)

    if install_requirements:
        model.install_requirements(error_level=error_level)
    else:
        model.ensure_requirements(error_level=error_level)

    config_dict = deepcopy(model.default_deployment_config_dict)
    model_path = model.get_path_in_dir(_MODELS_DIR)

    return fom.load_model(config_dict, model_path=model_path, **kwargs)


def find_model(name):
    """Returns the path to the model on disk.

    The model must be downloaded. Use :func:`download_model` to download
    models.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used

    Returns:
        the path to the model on disk

    Raises:
        ValueError: if the model does not exist or has not been downloaded
    """
    model, model_path = _get_model_in_dir(name)
    if not model.is_model_downloaded(model_path):
        raise ValueError("Model '%s' is not downloaded" % name)

    return model_path


def get_model(name):
    """Returns the ``eta.core.models.Model`` instance for the model with the
    given name.

    Args:
        name: the name of the model

    Returnsn ``eta.core.models.Model``:class:`ZooModel`
    """
    return _get_model(name)


def delete_model(name):
    """Deletes the model from local disk, if necessary.

    Args:
        name: the name of the model, which can have ``@<ver>`` appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is used
    """
    model, model_path = _get_model_in_dir(name)
    model.flush_model(model_path)


class HasBrainModel(etal.HasPublishedModel):
    """Mixin class for Config classes of :class:`fiftyone.core.models.Model`
    instances whose models are stored privately by the FiftyOne Brain.
    """

    def download_model_if_necessary(self):
        # pylint: disable=attribute-defined-outside-init
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided"
            )

        if self.model_path is None:
            self.model_path = download_model(self.model_name)

    @classmethod
    def _get_model(cls, model_name):
        return get_model(model_name)


def _load_models_manifest():
    return etam.ModelsManifest.from_json(_MODELS_MANIFEST_PATH)


def _get_model_in_dir(name):
    model = _get_model(name)
    model_path = model.get_path_in_dir(_MODELS_DIR)
    return model, model_path


def _get_model(name):
    if etam.Model.has_version_str(name):
        return _get_exact_model(name)

    return _get_latest_model(name)


def _get_exact_model(name):
    manifest = _load_models_manifest()
    try:
        return manifest.get_model_with_name(name)
    except etam.ModelError:
        raise ValueError("No model with name '%s' was found" % name)


def _get_latest_model(base_name):
    manifest = _load_models_manifest()
    try:
        return manifest.get_latest_model_with_base_name(base_name)
    except etam.ModelError:
        raise ValueError("No models found with base name '%s'" % base_name)
