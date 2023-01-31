"""
Simple configuration setup for these experiments

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import argparse


DATA_DIR = "/scratch/jason-data-cache/"
MODEL_DIR = "/scratch/jason-model-cache/"


class Config:
    """Encapsulates the configuration settings for a trial."""

    def __init__(self, d):
        """d is a dictionary of arguments from command-line."""

        self.batch_size = d["batch_size"]
        self.cold_start = d["cold_start"]
        self.epochs = d["epochs"]

        self.fixes = d["fixes"]
        self.make_fixes = True
        self.perfect_fixes = False
        if self.fixes == "no":
            self.makes_fixes = False
        if self.fixes == "perfect":
            self.make_fixes = True
            self.perfect_fixes = True

        self.model_path = d["model_path"]
        self.n_rounds = d["num_rounds"]
        self.n_max = d["num_max_samples"]
        self.p_corrupt = d["percent_corrupt"]
        self.p_fixable = d["percent_fixable"]
        self.p_initial = d["percent_initial"]

        self.runs = d["runs"]
        self.start_ipython = d["start_ipython"]
        self.stats_path = d["stats_path"]
        self.take = d["take"]

    def __str__(self):
        return str(vars(self))


def commandline():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=512,
        help="what is the batch size for the training iterations",
    )
    parser.add_argument(
        "--cold_start",
        "--cold-start",
        "-c",
        dest="cold_start",
        action="store_true",
    )
    parser.add_argument(
        "--no_cold_start",
        "--no-cold-start",
        "--no-cold_start",
        dest="cold_start",
        action="store_false",
    )
    parser.set_defaults(cold_start=False)
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=24,
        help="what is the number of epochs for each training iteration",
    )
    parser.add_argument(
        "--fixes",
        "-f",
        type=str,
        default="yes",
        help="make fixes? options are yes, no, and perfect",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        "--model-path",
        type=str,
        default=None,
        help="path to the file for loading/saving a model when applicable",
    )
    parser.add_argument(
        "--num_rounds",
        "--n_rounds",
        "--num-rounds",
        "--n-rounds",
        type=int,
        default=5,
        help="how many rounds to execute in a run",
    )
    parser.add_argument(
        "--num_max_samples",
        "--n_max_samples",
        "--num-max-samples",
        "--n-max-samples" "--num-max",
        "--num_max",
        "--n_max",
        "--n-max",
        type=int,
        default=-1,
        help="max samples to use in a run (default will use all samples)",
    )
    parser.add_argument(
        "--percent_corrupt",
        "--p_corrupt",
        "--percent-corrupt",
        "--p-corrupt",
        type=float,
        default=0.2,
        help="what percentage [0,1] of the data (annotations) to corrupt",
    )
    parser.add_argument(
        "--percent_fixable",
        "--p_fixable",
        "--percent-fixable",
        "--p-fixable",
        type=float,
        default=0.2,
        help="what percentage [0,1] of the data is fixable each round",
    )
    parser.add_argument(
        "--percent_initial",
        "--p_initial",
        "--percent-initial",
        "--p-initial",
        type=float,
        default=0.25,
        help="what percentage [0,1] of the data to use for initial training",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=6,
        help="how many times to run the trial for statistical purposes",
    )
    parser.add_argument(
        "--start_ipython",
        "--start-ipython",
        "-i",
        dest="start_ipython",
        action="store_true",
    )
    parser.add_argument(
        "--no_start_ipython",
        "--no-start-ipython",
        "--no-start_ipython",
        dest="start_ipython",
        action="store_false",
    )
    parser.set_defaults(start_ipython=False)
    parser.add_argument(
        "--stats_path",
        "-s",
        "--stats-path",
        type=str,
        default=None,
        help="path to the file for saving the statistics in json if desired",
    )
    parser.add_argument(
        "--take",
        "-t",
        type=int,
        default=None,
        help="how many samples to take in a subset of the dataset (default is all)",
    )

    args = parser.parse_args()

    return Config(args.__dict__)
