"""
le_test1 --> initial tests for using fiftyone with this code

This trains a cifar-10 under noisy labels with fixes via fiftyone.

After loading the data, a fraction of the annotations will be corrupted at
random.  Then, the training loop will ensue; after every training (full), a
fraction of the samples will be evaluated for erroneous annotations and fixed
(by using the original, correct ones)

What is the comparison?
To allow for a comparison, only half of the initial data will be used; at
each iteration through the main loop, a fraction of this data will be added.
For the annotation error case, we will fix the annotations and add the new
data.  For the baseline, we will just add the new data, which is what a naive
data scientist would do anyway.

TODO Another comparison would be to select just random samples to check and fix.

In addition, simple reporting on the accuracy of the identification of errors
in the annotations

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import argparse
from functools import partial
import json
import os
import random
import sys
import time

from scipy.misc import imsave
from scipy.stats import entropy

import fiftyone as fo
from fiftyone.core.odm import drop_database

from config import *
from datasets import *
from simple_resnet import *
from utils import Timer

TEMP_TRAIN_DIR="/tmp/le_test1/train"
TEMP_VALID_DIR="/tmp/le_test1/valid"

localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def write_images(root, images, overwrite=False):
    paths = []
    for i, s in enumerate(images):
        path = os.path.join(root, f"{i:05d}.jpg")
        paths.append(path)

        if overwrite or not os.path.exists(path):
            img = s.copy()
            img = transpose(img, source='CHW', target='HWC')
            imsave(path, img)

    return paths

def main(config):

    ## Initial Data Input
    # Produces train_set and valid_set that are lists of tuples: (image, label)
    timer = Timer()
    whole_dataset = cifar10(root=DATA_DIR)
    print("Preprocessing training data")
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'),
    ]
    whole_train_set = list(zip(*preprocess(whole_dataset['train'], [partial(pad, border=4)] + transforms).values()))
    valid_set = list(zip(*preprocess(whole_dataset['valid'], transforms).values()))
    print(f"Finished loading and preprocessing in {timer():.2f} seconds")

    print(f"train set: {len(whole_train_set)} samples")
    print(f"valid set: {len(valid_set)} samples")

    # TEMPORARY
    # Not production usage of fiftyone, but gets the point across to actually
    # get the Data into the system, from the format of this experiment
    #
    # The system does not support a good way to get data already loaded in
    # memory into it.  I think this is a significant limitation for certain
    # types of usage.  Furthermore, there are challenges here in this actual
    # usage of fiftyone because it is a bad idea to need to perform the
    # preprocessing on my data everytime I load an image in from disk, if I can
    # avoid that.
    #
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    os.makedirs(TEMP_VALID_DIR, exist_ok=True)
    train_image_paths = write_images(TEMP_TRAIN_DIR, list(zip(*whole_train_set))[0])
    valid_image_paths = write_images(TEMP_VALID_DIR, list(zip(*valid_set))[0])

    # make the actual labels for the cifar-10 world
    labels = []
    for i, s in enumerate(cifar10_classes):
        labels.append(fo.ClassificationLabel.create_new(label=s))

    timer = Timer()
    drop_database()
    dataset = fo.Dataset("le_cifar10")

    samples = []
    train_ids = []
    for i, s in enumerate(whole_train_set):
        sample = fo.Sample.create_new(train_image_paths[i], tags=["train"])
        sample.add_label("ground_truth", labels[s[1]])
        train_ids.append(dataset.add_sample(sample))

    #samples = []
    #for i, s in enumerate(whole_train_set):
    #    sample = fo.Sample.create_new(train_image_paths[i], tags=["train"])
    #    sample.add_label("ground_truth", labels[s[1]])
    #    samples.append(sample)
    #train_ids = dataset.add_samples(samples)

    #samples = []
    #for i, s in enumerate(valid_set):
    #    sample = fo.Sample.create_new(valid_image_paths[i], tags=["valid"])
    #    sample.add_label("ground_truth", labels[s[1]])
    #    samples.append(sample)
    #valid_ids = dataset.add_samples(samples)

    print(f"Finished getting data into fiftyone in {timer():.2f} seconds")

    sys.exit()

    # function of dataset
    N_labels = 10

    # set up the variables for training the model in each increment of the dataset size
    lr_schedule = PiecewiseLinear([0, 5, config.epochs], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    # compute the derived parameters for the trial based on the dataset and the
    # provided config.
    total_N = len(whole_train_set)

    # should be cleaner
    if(config.n_max < 0):
        config.n_max = total_N

    start_N = round(config.p_initial * total_N)

    incr_N = round((config.n_max-start_N) / config.n_increases)

    corrupt_N = round(config.p_corrupt * total_N)

    print(f'Setting up the experiment: {total_N} training samples.')
    print(f'- starting with {start_N}')
    print(f'- incrementing by {incr_N} for {config.n_increases} rounds')

    total_stats = {}

    for run_index in range(config.runs):

        print(f'Starting training iteration {run_index} at {localtime()}')

        # first, corrupt a random percentage of the data
        # this is wasteful and will create a copy of the training data in memory
        corr_train_set = whole_train_set.copy()

        # the strategy for corrupting the data randomly and then recalling the
        # original values
        corr_bool = [True for x in range(corrupt_N)]
        corr_bool.extend([False for x in range(total_N-corrupt_N)])

        for i in range(corrupt_N):
            t = corr_train_set[i]
            c = random.randint(0, N_labels-1)
            while c == t[1]:
                c = random.randint(0, N_labels-1)
            corr_train_set[i] = (t[0], c)

        shuffler = list(range(total_N))
        random.shuffle(shuffler)

        # common randomization of lists by way of an index permutation (shuffler)
        # original: a
        # shuffled: s
        # shuffler on indices: shuffler
        # s = [a[x] for x in shuffler]
        # supports --> s[i] = a[shuffler[i]]
        corr_train_set = [corr_train_set[x] for x in shuffler]
        corr_bool = [corr_bool[x] for x in shuffler]

        inuse_N = start_N

        model = Network(simple_resnet()).to(device).half()
        logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

        valid_batches = DataLoader(valid_set, config.batch_size, shuffle=False, drop_last=False)

        # initially randomly shuffle the dataset and take the initial number of samples
        corr_train_set_use = corr_train_set[0:inuse_N]
        corr_train_set_avail = corr_train_set[inuse_N:]
        print(f'Split training set into two; using {len(corr_train_set_use)}, available {len(corr_train_set_avail)}')

        sm = torch.nn.Softmax(dim=1)

        stats = {}

        for iteration in range(config.n_increases):
            print(f'beginning next round of training, using {inuse_N} samples')

            # uncomment this for a cold start to the model every iteration
            model = Network(simple_resnet()).to(device).half()
            logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

            train_batches = DataLoader(
                    Transform(corr_train_set_use, train_transforms),
                    config.batch_size, shuffle=True, set_random_choices=True, drop_last=True
            )
            lr = lambda step: lr_schedule(step/len(train_batches))/config.batch_size
            opts = [
                SGD(trainable_params(model).values(),
                {'lr': lr, 'weight_decay': Const(5e-4*config.batch_size), 'momentum': Const(0.9)})
            ]
            state[OPTS] = opts

            for epoch in range(config.epochs):
                logs.append(union({'epoch': epoch+1}, train_epoch(state, Timer(torch.cuda.synchronize), train_batches, valid_batches)))
            logs.df().query(f'epoch=={config.epochs}')[['train_acc', 'valid_acc']].describe()

            model.train(False) # == model.eval()

            # record scores for this iteration
            iteration_stats = {}
            iteration_stats["in_use"] = inuse_N

            correct = 0
            total = 0
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in valid_batches.dataloader:
                    images, labels = data
                    inputs = dict(input=images.cuda().half())
                    outputs = model(inputs)
                    y = outputs['logits']
                    _, predicted = torch.max(y, 1)
                    total += labels.size(0)
                    labels_gpu = labels.cuda().half()
                    correct += (predicted == labels_gpu).sum().item()
                    c = (predicted == labels_gpu).squeeze()
                    for i in range(min(config.batch_size, len(labels))):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            iteration_stats["validation_accuracy"] = correct / total

            # how many can we fix?
            fix_N = round(inuse_N * config.p_fixable)

            if config.perfect_fixes:
                # implements an upper-bound-like stat, that only selects incorrect samples
                # assumes `config.make_fixes = True`
                fixed = 0

                for fix_i, needs_fixing in enumerate(corr_bool[:inuse_N]):
                    if not needs_fixing:
                        continue
                    t = corr_train_set_use[fix_i]
                    w = whole_train_set[shuffler[fix_i]][1]
                    corr_train_set_use[fix_i] = (t[0], w)
                    corr_bool[fix_i] = False
                    fixed += 1
                    if fixed >= fix_N:
                        break

                iteration_stats["fix_correct"] = fixed
                iteration_stats["fix_percent_of_N"] = fixed / fix_N
                current_corrupted = sum(corr_bool[:inuse_N])
                iteration_stats["current_corrupted"] = current_corrupted

                print(f'"predicted" {fixed} of {fix_N} proposals to fix correctly')
                print(f'{fixed / fix_N}')
                print(f'there are currently a total of {current_corrupted} corrupted labels in use')
            else:
                review_batches = DataLoader(corr_train_set_use, 1, shuffle=False, drop_last=False)
                scores = []
                with torch.no_grad():
                    for data in review_batches.dataloader:
                        images, labels = data
                        inputs = dict(input=images.cuda().half())
                        outputs = model(inputs)
                        y = outputs['logits']
                        #_, predicted = torch.max(y, 1)

                        # likelihood of error in annotation can be a number of
                        # different heuristics.  just use entropy for now to get the
                        # whole script going

                        # todo: implement an entropy in torch to work on gpu
                        z = sm(y).cpu().numpy().squeeze()
                        #e  = entropy(sm(y).numpy())
                        e  = entropy(z)
                        scores.append(e)

                scores = np.asarray(scores)
                si = np.argsort(scores)
                si = np.flip(si.squeeze()) # because higher entropy at front
                                           # beware of a need to sort in other
                                           # heuristics

                # how many of those that we want to fix need to be fixed
                # this is a bit of a funky calculation because we may want to fix more
                # than we need to if we've already fixed some --> so, just count the
                # true positives for now and neglect any false positives
                fix_correct = 0
                for fix_i in si[:fix_N]:
                    if corr_bool[fix_i]:
                        fix_correct += 1

                iteration_stats["fix_correct"] = fix_correct
                iteration_stats["fix_percent_of_N"] = fix_correct / fix_N

                current_corrupted = sum(corr_bool[:inuse_N])
                iteration_stats["current_corrupted"] = current_corrupted

                print(f'predicted {fix_correct} of {fix_N} proposals to fix correctly')
                print(f'{fix_correct / fix_N}')
                print(f'there are currently a total of {current_corrupted} corrupted labels in use')

                if config.make_fixes:
                    print(f'making the fixes')
                    # fix the corruptions
                    for fix_i in si[:fix_N]:
                        t = corr_train_set_use[fix_i]
                        w = whole_train_set[shuffler[fix_i]][1]
                        corr_train_set_use[fix_i] = (t[0], w)
                        corr_bool[fix_i] = False
                else:
                    print(f'not making the fixes because of the config')

            model.train(True)

            # extend the corr_train_set_use with that from avail
            corr_train_set_use.extend(corr_train_set_avail[0:incr_N])
            corr_train_set_avail = corr_train_set_avail[incr_N:]
            inuse_N += incr_N
            assert inuse_N == len(corr_train_set_use)

            stats[inuse_N] = iteration_stats

        total_stats[run_index] = stats

    print(f'finished the full training; stats to follow')
    print(total_stats)

    if config.stats_path:
        with open(config.stats_path, 'w') as fp:
            json.dump(total_stats, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="labelerror", add_help=True)

    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=512,
        help="what is the batch size for the training iterations")
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=24,
        help="what is the number of epochs for each training iteration")
    parser.add_argument(
        "--fixes", "-f",
        type=str,
        default="yes",
        help="make fixes? options are yes, no, and perfect")
    parser.add_argument(
        "--num_increases", "--n_increases", "-i",
        type=int,
        default=5,
        help="how many rounds to execute in a run")
    parser.add_argument(
        "--num_max_samples", "--n_max_samples", "--num-max", "--n_max",
        type=int,
        default=-1,
        help="max samples to use in a run (default will use all samples)")
    parser.add_argument(
        "--percent_corrupt", "--p_corrupt",
        type=float,
        default=0.2,
        help="what percentage [0,1] of the data (annotations) to corrupt")
    parser.add_argument(
        "--percent_fixable", "--p_fixable",
        type=float,
        default=0.2,
        help="what percentage [0,1] of the data is fixable each round")
    parser.add_argument(
        "--percent_initial", "--p_initial",
        type=float,
        default=0.25,
        help="what percentage [0,1] of the data to use for initial training")
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=6,
        help="how many times to run the trial for statistical purposes")
    parser.add_argument(
        "--stats_path", "-s",
        type=str,
        default=None,
        help="path to the file for saving the statistics in json if desired")

    args = parser.parse_args()

    config = Config(args.__dict__)

    print(f"running with config: {config}")

    main(config)
