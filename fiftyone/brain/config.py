"""
Brain config.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from fiftyone.core.config import EnvConfig


class BrainConfig(EnvConfig):
    """FiftyOne brain configuration settings."""

    _BUILTIN_SIMILARITY_BACKENDS = {
        "sklearn": {
            "config_cls": "fiftyone.brain.internal.core.sklearn.SklearnSimilarityConfig",
        },
        "pinecone": {
            "config_cls": "fiftyone.brain.internal.core.pinecone.PineconeSimilarityConfig",
        },
    }

    def __init__(self, d=None):
        if d is None:
            d = {}

        self.default_similarity_backend = self.parse_string(
            d,
            "default_similarity_backend",
            env_var="FIFTYONE_BRAIN_DEFAULT_SIMILARITY_BACKEND",
            default="sklearn",
        )

        self.similarity_backends = self._parse_similarity_backends(d)

    def _parse_similarity_backends(self, d):
        d = d.get("similarity_backends", {})
        env_vars = dict(os.environ)

        #
        # `FIFTYONE_BRAIN_SIMILARITY_BACKENDS` can be used to declare which
        # backends are exposed. This may exclude builtin backends and/or
        # declare new backends
        #

        if "FIFTYONE_BRAIN_SIMILARITY_BACKENDS" in env_vars:
            backends = env_vars["FIFTYONE_BRAIN_SIMILARITY_BACKENDS"].split(
                ","
            )

            # Declare new backends and omit any others not in `backends`
            d = {backend: d.get(backend, {}) for backend in backends}
        else:
            backends = sorted(self._BUILTIN_SIMILARITY_BACKENDS.keys())

            # Declare builtin backends if necessary
            for backend in backends:
                if backend not in d:
                    d[backend] = {}

        #
        # Extract parameters from any environment variables of the form
        # `FIFTYONE_BRAIN_SIMILARITY_<BACKEND>_<PARAMETER>`
        #

        for backend, parameters in d.items():
            prefix = "FIFTYONE_BRAIN_SIMILARITY_%s_" % backend.upper()
            for env_name, env_value in env_vars.items():
                if env_name.startswith(prefix):
                    name = env_name[len(prefix) :].lower()
                    value = _parse_env_value(env_value)
                    parameters[name] = value

        #
        # Set default parameters for builtin similarity backends
        #

        for backend, defaults in self._BUILTIN_SIMILARITY_BACKENDS.items():
            if backend not in d:
                continue

            d_backend = d[backend]
            for name, value in defaults.items():
                if name not in d_backend:
                    d_backend[name] = value

        return d


def locate_brain_config():
    """Returns the path to the :class:`BrainConfig` on disk.

    The default location is ``~/.fiftyone/brain_config.json``, but you can
    override this path by setting the ``FIFTYONE_BRAIN_CONFIG_PATH``
    environment variable.

    Note that a config file may not actually exist on disk in the default
    location, in which case the default config settings will be used.

    Returns:
        the path to the :class:`BrainConfig` on disk

    Raises:
        OSError: if the brain config path has been customized but the file does
            not exist on disk
    """
    if "FIFTYONE_BRAIN_CONFIG_PATH" not in os.environ:
        return os.path.join(
            os.path.expanduser("~"), ".fiftyone", "brain_config.json"
        )

    config_path = os.environ["FIFTYONE_BRAIN_CONFIG_PATH"]
    if not os.path.isfile(config_path):
        raise OSError("Brain config file '%s' not found" % config_path)

    return config_path


def load_brain_config():
    """Loads the FiftyOne brain config.

    Returns:
        a :class:`BrainConfig` instance
    """
    brain_config_path = locate_brain_config()
    if os.path.isfile(brain_config_path):
        return BrainConfig.from_json(brain_config_path)

    return BrainConfig()


def _parse_env_value(value):
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    if value in ("True", "true"):
        return True

    if value in ("False", "false"):
        return False

    if value in ("None", ""):
        return None

    if "," in value:
        return [_parse_env_value(v) for v in value.split(",")]

    return value