"""
Utilities for experiments.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import pandas as pd
import time

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


class Timer:
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


sep = "/"


def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]


def normpath(path):
    # simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == "..":
            parts.pop()
        elif p.startswith(sep):
            parts = [p]
        else:
            parts.append(p)
    return sep.join(parts)


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


def map_nested(func, nested_dict):
    return {
        k: map_nested(func, v) if isinstance(v, dict) else func(v)
        for k, v in nested_dict.items()
    }


def group_by_key(items):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return res


default_table_formats = {
    float: "{:{w}.4f}",
    str: "{:>{w}s}",
    "default": "{:{w}}",
    "title": "{:>{w}s}",
}


def table_formatter(val, is_title=False, col_width=12, formats=None):
    formats = formats or default_table_formats
    type_ = (
        lambda val: float if isinstance(val, (float, np.float)) else type(val)
    )
    return (
        formats["title"]
        if is_title
        else formats.get(type_(val), formats["default"])
    ).format(val, w=col_width)


def every(n, col):
    return lambda data: data[col] % n == 0


class Table:
    def __init__(
        self, keys=None, report=(lambda data: True), formatter=table_formatter
    ):
        self.keys, self.report, self.formatter = keys, report, formatter
        self.log = []

    def append(self, data):
        self.log.append(data)
        data = {" ".join(p): v for p, v in path_iter(data)}
        self.keys = self.keys or data.keys()
        if len(self.log) is 1:
            print(*(self.formatter(k, True) for k in self.keys))
        if self.report(data):
            print(*(self.formatter(data[k]) for k in self.keys))

    def df(self):
        return pd.DataFrame(
            [{"_".join(p): v for p, v in path_iter(row)} for row in self.log]
        )
