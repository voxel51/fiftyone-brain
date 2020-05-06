"""
Simple configuration setup for these experiments

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
DATA_DIR="/scratch/jason-data-cache/"
MODEL_DIR="/scratch/jason-model-cache/"

class Config:
    '''Encapsulates the configuration settings for a trial.'''

    def __init__(self, d):
        '''d is a dictionary of arguments from command-line.'''

        self.batch_size = d["batch_size"]
        self.epochs = d["epochs"]

        self.fixes = d["fixes"]
        self.make_fixes = True
        self.perfect_fixes = False
        if self.fixes == "no":
            self.makes_fixes = False
        if self.fixes == "perfect":
            self.make_fixes = True
            self.perfect_fixes = True

        self.n_increases = d["num_increases"]
        self.n_max = d["num_max_samples"]
        self.p_corrupt = d["percent_corrupt"]
        self.p_fixable = d["percent_fixable"]
        self.p_initial = d["percent_initial"]

        self.runs = d["runs"]
        self.stats_path = d["stats_path"]

    def __str__(self):
        return str(vars(self))
