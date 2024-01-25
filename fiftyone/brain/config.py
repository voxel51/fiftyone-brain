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
        "qdrant": {
            "config_cls": "fiftyone.brain.internal.core.qdrant.QdrantSimilarityConfig",
        },
        "milvus": {
            "config_cls": "fiftyone.brain.internal.core.milvus.MilvusSimilarityConfig",
        },
        "lancedb": {
            "config_cls": "fiftyone.brain.internal.core.lancedb.LanceDBSimilarityConfig",
        },
        "redis": {
            "config_cls": "fiftyone.brain.internal.core.redis.RedisSimilarityConfig",
        },
        "mongodb": {
            "config_cls": "fiftyone.brain.internal.core.mongodb.MongoDBSimilarityConfig",
        },
    }

    _BUILTIN_VISUALIZATION_METHODS = {
        "umap": {
            "config_cls": "fiftyone.brain.visualization.UMAPVisualizationConfig",
        },
        "tsne": {
            "config_cls": "fiftyone.brain.visualization.TSNEVisualizationConfig",
        },
        "pca": {
            "config_cls": "fiftyone.brain.visualization.PCAVisualizationConfig",
        },
        "manual": {
            "config_cls": "fiftyone.brain.visualization.ManualVisualizationConfig",
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
        if self.default_similarity_backend not in self.similarity_backends:
            self.default_similarity_backend = next(
                iter(sorted(self.similarity_backends.keys())), None
            )

        self.default_visualization_method = self.parse_string(
            d,
            "default_visualization_method",
            env_var="FIFTYONE_BRAIN_DEFAULT_VISUALIZATION_METHOD",
            default="umap",
        )

        self.visualization_methods = self._parse_visualization_methods(d)
        if self.default_visualization_method not in self.visualization_methods:
            self.default_visualization_method = next(
                iter(sorted(self.visualization_methods.keys())), None
            )

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

            # Special syntax to append rather than override default backends
            if "*" in backends:
                backends = set(b for b in backends if b != "*")
                backends |= set(self._BUILTIN_SIMILARITY_BACKENDS.keys())

            d = {backend: d.get(backend, {}) for backend in backends}
        else:
            backends = self._BUILTIN_SIMILARITY_BACKENDS.keys()
            for backend in backends:
                if backend not in d:
                    d[backend] = {}

        #
        # Extract parameters from any environment variables of the form
        # `FIFTYONE_BRAIN_SIMILARITY_<BACKEND>_<PARAMETER>`
        #

        for backend, d_backend in d.items():
            prefix = "FIFTYONE_BRAIN_SIMILARITY_%s_" % backend.upper()
            for env_name, env_value in env_vars.items():
                if env_name.startswith(prefix):
                    name = env_name[len(prefix) :].lower()
                    value = _parse_env_value(env_value)
                    d_backend[name] = value

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

    def _parse_visualization_methods(self, d):
        d = d.get("visualization_methods", {})
        env_vars = dict(os.environ)

        #
        # `FIFTYONE_BRAIN_VISUALIZATION_METHODS` can be used to declare which
        # methods are exposed. This may exclude builtin methods and/or declare
        # new methods
        #

        if "FIFTYONE_BRAIN_VISUALIZATION_METHODS" in env_vars:
            methods = env_vars["FIFTYONE_BRAIN_VISUALIZATION_METHODS"].split(
                ","
            )

            # Special syntax to append rather than override default methods
            if "*" in methods:
                methods = set(m for m in methods if m != "*")
                methods |= set(self._BUILTIN_VISUALIZATION_METHODS.keys())

            d = {method: d.get(method, {}) for method in methods}
        else:
            methods = self._BUILTIN_VISUALIZATION_METHODS.keys()
            for method in methods:
                if method not in d:
                    d[method] = {}

        #
        # Extract parameters from any environment variables of the form
        # `FIFTYONE_BRAIN_VISUALIZATION_<METHOD>_<PARAMETER>`
        #

        for method, d_method in d.items():
            prefix = "FIFTYONE_BRAIN_VISUALIZATION_%s_" % method.upper()
            for env_name, env_value in env_vars.items():
                if env_name.startswith(prefix):
                    name = env_name[len(prefix) :].lower()
                    value = _parse_env_value(env_value)
                    d_method[name] = value

        #
        # Set default parameters for builtin visualization methods
        #

        for method, defaults in self._BUILTIN_VISUALIZATION_METHODS.items():
            if method not in d:
                continue

            d_method = d[method]
            for name, value in defaults.items():
                if name not in d_method:
                    d_method[name] = value

        return d


def locate_brain_config():
    """Returns the path to the :class:`BrainConfig` on disk.

    The default location is ``~/.fiftyone/brain_config.json``, but you can
    override this path by setting the ``FIFTYONE_BRAIN_CONFIG_PATH``
    environment variable.

    Note that a config file may not actually exist on disk.

    Returns:
        the path to the :class:`BrainConfig` on disk
    """
    if "FIFTYONE_BRAIN_CONFIG_PATH" not in os.environ:
        return os.path.join(
            os.path.expanduser("~"), ".fiftyone", "brain_config.json"
        )

    return os.environ["FIFTYONE_BRAIN_CONFIG_PATH"]


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
