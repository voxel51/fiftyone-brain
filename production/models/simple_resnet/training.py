"""
Training functions

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import namedtuple
from functools import partial, singledispatch
from itertools import chain

import numpy as np
import torch

from fiftyone.brain.internal.models.simple_resnet import *

from utils import *


@singledispatch
def to_numpy(x):
    raise NotImplementedError


@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()


## Losses, Optimizers


class CrossEntropyLoss(namedtuple("CrossEntropyLoss", [])):
    def __call__(self, log_probs, target):
        return torch.nn.functional.nll_loss(
            log_probs, target, reduction="none"
        )


class KLLoss(namedtuple("KLLoss", [])):
    def __call__(self, log_probs):
        return -log_probs.mean(dim=1)


class Correct(namedtuple("Correct", [])):
    def __call__(self, classifier, target):
        return classifier.max(dim=1)[1] == target


class LogSoftmax(namedtuple("LogSoftmax", ["dim"])):
    def __call__(self, x):
        return torch.nn.functional.log_softmax(x, self.dim, _stacklevel=5)


x_ent_loss = Network(
    {
        "loss": (nn.CrossEntropyLoss(reduction="none"), ["logits", "target"]),
        "acc": (Correct(), ["logits", "target"]),
    }
)

label_smoothing_loss = lambda alpha: Network(
    {
        "logprobs": (LogSoftmax(dim=1), ["logits"]),
        "KL": (KLLoss(), ["logprobs"]),
        "xent": (CrossEntropyLoss(), ["logprobs", "target"]),
        "loss": (AddWeighted(wx=1 - alpha, wy=alpha), ["xent", "KL"]),
        "acc": (Correct(), ["logits", "target"]),
    }
)

trainable_params = lambda model: (
    {k: p for k, p in model.named_parameters() if p.requires_grad}
)


def nesterov_update(w, dw, v, lr, weight_decay, momentum):
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))


norm = lambda x: torch.norm(x.reshape(x.size(0), -1).float(), dim=1)[
    :, None, None, None
]


def LARS_update(w, dw, v, lr, weight_decay, momentum):
    nesterov_update(
        w,
        dw,
        v,
        lr * (norm(w) / (norm(dw) + 1e-2)).to(w.dtype),
        weight_decay,
        momentum,
    )


def zeros_like(weights):
    return [torch.zeros_like(w) for w in weights]


def optimiser(weights, param_schedule, update, state_init):
    weights = list(weights)
    return {
        "update": update,
        "param_schedule": param_schedule,
        "step_number": 0,
        "weights": weights,
        "opt_state": state_init(weights),
    }


def opt_step(update, param_schedule, step_number, weights, opt_state):
    step_number += 1
    param_values = {k: f(step_number) for k, f in param_schedule.items()}
    for w, v in zip(weights, opt_state):
        if w.requires_grad:
            update(w.data, w.grad.data, v, **param_values)

    return {
        "update": update,
        "param_schedule": param_schedule,
        "step_number": step_number,
        "weights": weights,
        "opt_state": opt_state,
    }


LARS = partial(optimiser, update=LARS_update, state_init=zeros_like)
SGD = partial(optimiser, update=nesterov_update, state_init=zeros_like)


## Training Code
LOSS = "loss"
OPTS = "optimisers"
ACT_LOG = "activation_log"


def reduce(batches, state, steps):
    # state: is a dictionary
    # steps: are functions that take (batch, state)
    # and return a dictionary of updates to the state (or None)

    for batch in chain(batches, [None]):
        # we send an extra batch=None at the end for steps that
        # need to do some tidying-up (e.g. log_activations)
        for step in steps:
            updates = step(batch, state)
            if updates:
                for k, v in updates.items():
                    state[k] = v

    return state


def forward(training_mode):
    def step(batch, state):
        if not batch:
            return

        model = (
            state[MODEL]
            if training_mode or (VALID_MODEL not in state)
            else state[VALID_MODEL]
        )
        if model.training != training_mode:  # without the guard it's slow!
            model.train(training_mode)

        return {OUTPUT: state[LOSS](model(batch))}

    return step


def backward(dtype=None):
    def step(batch, state):
        state[MODEL].zero_grad()
        if not batch:
            return

        loss = state[OUTPUT][LOSS]
        if dtype is not None:
            loss = loss.to(dtype)

        loss.sum().backward()

    return step


def opt_steps(batch, state):
    if not batch:
        return

    return {OPTS: [opt_step(**opt) for opt in state[OPTS]]}


def log_activations(node_names=("loss", "acc")):
    def step(batch, state):
        if "_tmp_logs_" not in state:
            state["_tmp_logs_"] = []
        if batch:
            state["_tmp_logs_"].extend(
                (k, state[OUTPUT][k].detach()) for k in node_names
            )
        else:
            res = {
                k: to_numpy(torch.cat(xs)).astype(np.float)
                for k, xs in group_by_key(state["_tmp_logs_"]).items()
            }
            del state["_tmp_logs_"]
            return {ACT_LOG: res}

    return step


epoch_stats = lambda state: {k: np.mean(v) for k, v in state[ACT_LOG].items()}

default_train_steps = (
    forward(training_mode=True),
    log_activations(("loss", "acc")),
    backward(),
    opt_steps,
)
default_valid_steps = (
    forward(training_mode=False),
    log_activations(("loss", "acc")),
)


def train_epoch(
    state,
    timer,
    train_batches,
    valid_batches,
    train_steps=default_train_steps,
    valid_steps=default_valid_steps,
    on_epoch_end=(lambda state: state),
):
    train_summary, train_time = (
        epoch_stats(on_epoch_end(reduce(train_batches, state, train_steps))),
        timer(),
    )
    valid_summary, valid_time = (
        epoch_stats(reduce(valid_batches, state, valid_steps)),
        timer(include_in_total=False),
    )  # DAWNBench rules
    return {
        "train": union({"time": train_time}, train_summary),
        "valid": union({"time": valid_time}, valid_summary),
        "total time": timer.total_time,
    }
