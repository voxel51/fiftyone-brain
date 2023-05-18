"""
Script to plot a scalar value that was computed as certain points in an
iterative process.  It will average results from many such runs and plot shaded
error bars

It can also plot data-files that have multiple fields of results and select one
of them to plot per file.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


class Config:
    def __init__(self, d):
        '''d is a dictionary of arguments from command-line.'''
        self.colors = list(plt.get_cmap('Set1').colors)
        self.colors_N = len(self.colors)
        self.error_alpha = 0.2

        self.field = d['field']
        self.title = d['title']
        self.xlabel = d['xlabel']
        self.ylabel = d['ylabel']

    def color(self, index):
        '''Retrieve the color tuple for index with wrap-around.'''
        return self.colors[index % self.colors_N]


def read_to_array(config, json_path):
    '''Read the data in the file at json_path into a numpy array.

    The rows in the array are the experiment results and the columns are the
    values.  Returns a tuple with the data, indices where indices is a
    vector of the sample locations for the columns.
    '''
    with open(json_path, "rt") as f:
        d = json.load(f)

    num_rows = len(d.keys())

    assert num_rows >= 0

    indices = np.asarray(np.sort(list(d[next(iter(d.keys()))].keys())))
    ind_lut = {}
    for i, ind in enumerate(indices):
        ind_lut[ind] = i

    num_cols = len(indices)

    data = np.zeros((num_rows, num_cols))

    for i, (k, v) in enumerate(d.items()):
        # v is itself a dictionary with the keys being the indices and the
        # values being the actual graph values
        ind_check = np.sort(list(v.keys()))
        assert np.array_equal(ind_check, indices)

        if not isinstance(v[indices[0]], dict):
            # build the values this way in case they were not ordered in the dict
            for j, index in enumerate(indices):
                data[i][ind_lut[index]] = v[index]
        else:
            if not config.field:
                print(f'fail: data file has dicts but no field specified')
            for j, index in enumerate(indices):
                data[i][ind_lut[index]] = v[index][config.field]

    return indices, data


def main_plot(config, list_of_inputs):
    '''Main driver for the graphing, each item in the list_of_inputs is a path
    to a json file with a set of runs to plot (and, optionally a name).

    name is specified in one of two ways:
    1.  If the command-line arg has just the json file: "./foo.json", then we
    will derive the name as "foo".
    2.  Using a colon: foobar:./foo.json and the name here is "foobar".
    '''

    fig, ax = plt.subplots()

    for i, input in enumerate(list_of_inputs):
        print(f"Working on {os.path.basename(input)}")

        # check if input has a colon and if so, split on it
        if ":" in input:
            name, json = input.split(":")
        else:
            json = input
            name = os.path.splitext(os.path.basename(input))[0]

        indices, data = read_to_array(config, json)

        mean = data.mean(0)
        std = data.std(0)

        line, =  ax.plot(indices, mean, color=config.color(i))
        ax.fill_between(indices, mean - std, mean + std, color=config.color(i),
                        alpha=config.error_alpha)

        line.set_label(name)

    plt.title(config.title)
    plt.xlabel(config.xlabel)
    plt.ylabel(config.ylabel)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="plot_run_scalar", add_help=True)

    parser.add_argument("inputs", nargs="+",
                        help="the input json files with one per curve")
    parser.add_argument("--field", "-f", type=str, default=None,
                        help="if data is a dictionary, use this key to get values for plotting")
    parser.add_argument("--xlabel", "-x", type=str, default=None,
                        help="string to use as the label of the x-axis.")
    parser.add_argument("--ylabel", "-y", type=str, default=None,
                        help="string to use as the label of the y-axis.")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="string to use as the title.")

    args = parser.parse_args()

    config = Config(args.__dict__)
    main_plot(config, args.inputs)
