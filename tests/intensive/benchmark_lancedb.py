"""
Add-cost and query-latency benchmark for the LanceDB similarity backend.

Drives the connector's own write path over a row-count sweep, so add cost can
be read against table size. Tables are seeded through LanceDB directly rather
than through the connector: seeding a million rows one ``add_to_index`` batch
at a time would dominate the run, and the measurement wanted here is the cost
of one incremental add against a table that is already large.

Vectors are uniform random, which is fine for write cost and for exhaustive
query cost, since neither depends on how the vectors are distributed. It is
not a valid setting for measuring an ANN index, whose recall depends entirely
on the data having cluster structure.

Usage::

    python tests/intensive/benchmark_lancedb.py
    python tests/intensive/benchmark_lancedb.py --sizes 10000 --dims 2048

The defaults -- 1k/100k/1M rows, 512 dimensions, 100-row batches, 20 timed
repeats -- are the settings the recorded numbers came from.

Requires ``pip install lancedb`` and a running MongoDB, since the index is
constructed against a throwaway dataset.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import argparse
import os
import shutil
import statistics
import tempfile
import time

import numpy as np

import lancedb

import fiftyone as fo

from fiftyone.brain.internal.core.lancedb import (
    LanceDBSimilarityConfig,
    LanceDBSimilarityIndex,
    _to_arrow_table,
)

_DATASET_NAME = "lancedb-benchmark"
_TABLE_NAME = "benchmark"
_SEED_BATCH_SIZE = 50000

# Discarded before timing: these pay LanceDB's process-wide start-up and build
# the scalar index, costs that would otherwise land on whichever size runs
# first and read as growth
_WARMUP_REPEATS = 10

# Rows per bulk add while growing the tail to a checkpoint. Large so growing
# is cheap; the timed adds use `batch_size` so they stay comparable.
_TAIL_GROW_BATCH = 5000


def _random_embeddings(num_rows, dims, rng):
    return rng.random((num_rows, dims), dtype=np.float32)


def _batch_ids(tag, repeat, count):
    return ["%s-%d-%06d" % (tag, repeat, i) for i in range(count)]


def _seed_table(uri, num_rows, dims, rng):
    """Writes ``num_rows`` rows straight to LanceDB, bypassing the connector.

    Args:
        uri: the database URI
        num_rows: the number of rows to write
        dims: the embedding dimension
        rng: a ``numpy.random.Generator``

    Returns:
        the number of rows written
    """
    db = lancedb.connect(uri)
    table = None

    for start in range(0, num_rows, _SEED_BATCH_SIZE):
        stop = min(start + _SEED_BATCH_SIZE, num_rows)
        ids = ["seed-%09d" % i for i in range(start, stop)]
        embeddings = _random_embeddings(stop - start, dims, rng)
        rows = _to_arrow_table(ids, ids, embeddings)

        if table is None:
            table = db.create_table(_TABLE_NAME, rows, mode="overwrite")
        else:
            table.add(rows)

    return table.count_rows()


def _make_index(samples, uri):
    """Opens an index over ``uri``.

    ``samples=None`` builds one without binding it to a sample collection,
    which skips the database entirely. The write paths reach only the table,
    the connection and the config, so add, remove and the id index all work
    unbound -- but anything reading the view does not, so a query needs the
    bound form.
    """
    config = LanceDBSimilarityConfig(table_name=_TABLE_NAME, uri=uri)
    if samples is None:
        index = LanceDBSimilarityIndex.__new__(LanceDBSimilarityIndex)
        index._config = config
        index._initialize()
        return index

    return LanceDBSimilarityIndex(samples, config, "benchmark")


def _time_adds(index, rng, *, batch_size, dims, repeats, tag="add"):
    """Times ``repeats`` adds of ``batch_size`` rows each, in milliseconds."""
    timings = []

    for repeat in range(repeats):
        ids = np.array(_batch_ids(tag, repeat, batch_size))
        embeddings = _random_embeddings(batch_size, dims, rng)

        start = time.perf_counter()
        index.add_to_index(embeddings, ids, reload=False)
        timings.append((time.perf_counter() - start) * 1000)

    return timings


def _time_queries(index, rng, *, dims, k, repeats):
    """Times ``repeats`` unfiltered k-NN queries, in milliseconds."""
    timings = []

    for _ in range(repeats):
        query = _random_embeddings(1, dims, rng)[0]

        start = time.perf_counter()
        index._kneighbors(query=query, k=k)
        timings.append((time.perf_counter() - start) * 1000)

    return timings


def _time_removes(index, ids):
    """Times the removal of each ID in turn, in milliseconds."""
    timings = []

    for _id in ids:
        start = time.perf_counter()
        index.remove_from_index(sample_ids=[_id], reload=False)
        timings.append((time.perf_counter() - start) * 1000)

    return timings


def _unindexed_tail(index):
    """Rows written since the ``id`` index was last built.

    A merge probes the index and then scans this tail, so add cost rises with
    it until ``optimize()`` folds the tail in.
    """
    for config in index.table.list_indices():
        if config.columns == ["id"]:
            return config.num_unindexed_rows

    return 0


def _num_fragments(index):
    """Fragments backing the table.

    Every add appends one. Query cost rises with the count while merge and
    delete are largely unaffected, so an incrementally built table reads more
    slowly than a bulk-loaded one holding the same rows.
    """
    return index.table.stats()["fragment_stats"]["num_fragments"]


def _summarize(label, timings):
    """Prints the timing spread and returns the median."""
    median = statistics.median(timings)
    print(
        "    %-8s median %7.1f ms   min %7.1f ms   max %7.1f ms"
        % (label, median, min(timings), max(timings))
    )
    return median


def _run_size(samples, num_rows, *, dims, batch_size, repeats, k, seed):
    rng = np.random.default_rng(seed)
    uri = tempfile.mkdtemp(prefix="lancedb-benchmark-")

    try:
        print("  seeding %d rows at %d dims..." % (num_rows, dims))
        seeded = _seed_table(uri, num_rows, dims, rng)

        index = _make_index(samples, uri)
        assert index.total_index_size == seeded, "seeded table did not open"

        _time_adds(
            index,
            rng,
            batch_size=batch_size,
            dims=dims,
            repeats=_WARMUP_REPEATS,
            tag="warmup",
        )
        _time_queries(index, rng, dims=dims, k=k, repeats=_WARMUP_REPEATS)
        for repeat in range(_WARMUP_REPEATS):
            index.remove_from_index(
                sample_ids=_batch_ids("warmup", repeat, batch_size),
                reload=False,
            )

        add_ms = _summarize(
            "add",
            _time_adds(
                index,
                rng,
                batch_size=batch_size,
                dims=dims,
                repeats=repeats,
            ),
        )
        query_ms = _summarize(
            "query", _time_queries(index, rng, dims=dims, k=k, repeats=repeats)
        )
        # Removes rows the timed adds just wrote; batch 0 holds `batch_size`
        # of them, so asking for more would time deletes of IDs that were
        # never written and report them as removal cost
        removed_ids = _batch_ids("add", 0, min(repeats, batch_size))
        remove_ms = _summarize("remove", _time_removes(index, removed_ids))
        print(
            "    %-8s %d unindexed rows, %d fragments"
            % ("state", _unindexed_tail(index), _num_fragments(index))
        )

        return add_ms, query_ms, remove_ms
    finally:
        shutil.rmtree(uri, ignore_errors=True)


def _run_tail_sweep(
    samples,
    num_rows,
    *,
    dims,
    batch_size,
    repeats,
    seed,
    checkpoints,
    grow_batch,
):
    """Times adds against a growing unindexed tail.

    The size sweep holds the tail near zero -- it writes `repeats` batches and
    stops -- so its flat add cost is flat in table size at a tail of a couple
    of thousand rows, which says nothing about what happens as the tail grows.
    `_ensure_id_index` rebuilds only when the index is absent, never when it is
    merely stale, so in service the tail grows until something calls
    ``optimize()``.

    At each checkpoint the tail is grown in bulk, then add cost is measured at
    the same `batch_size` used everywhere else so the numbers are comparable.
    The timed adds grow the tail themselves, by `repeats * batch_size` rows;
    the tail is reported after them.
    """
    rng = np.random.default_rng(seed)
    uri = tempfile.mkdtemp(prefix="lancedb-tail-")

    try:
        print("  seeding %d rows at %d dims..." % (num_rows, dims))
        _seed_table(uri, num_rows, dims, rng)
        index = _make_index(samples, uri)

        # Builds the id index, so every later add lands in the tail
        _time_adds(
            index,
            rng,
            batch_size=batch_size,
            dims=dims,
            repeats=_WARMUP_REPEATS,
            tag="warmup",
        )

        print(
            "    %10s %12s %10s %12s"
            % ("target", "tail after", "add median", "fragments")
        )
        grown = 0
        for target in checkpoints:
            while grown < target:
                step = min(grow_batch, target - grown)
                index.add_to_index(
                    _random_embeddings(step, dims, rng),
                    np.array(_batch_ids("grow", grown, step)),
                    reload=False,
                )
                grown += step

            add_ms = statistics.median(
                _time_adds(
                    index,
                    rng,
                    batch_size=batch_size,
                    dims=dims,
                    repeats=repeats,
                    tag="tail%d" % target,
                )
            )
            print(
                "    %10d %12d %7.1f ms %12d"
                % (
                    target,
                    _unindexed_tail(index),
                    add_ms,
                    _num_fragments(index),
                )
            )
            grown += repeats * batch_size

        # What it costs to fold the tail back in, which is the other half of
        # any reindex cadence
        start = time.perf_counter()
        index.table.optimize()
        optimize_ms = (time.perf_counter() - start) * 1000
        after = statistics.median(
            _time_adds(
                index,
                rng,
                batch_size=batch_size,
                dims=dims,
                repeats=repeats,
                tag="post",
            )
        )
        print(
            "    optimize() took %.0f ms; tail now %d, add back to %.1f ms"
            % (optimize_ms, _unindexed_tail(index), after)
        )
    finally:
        shutil.rmtree(uri, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("Usage::")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        default="1000,100000,1000000",
        help="comma-separated table sizes to sweep",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=512,
        help="embedding dimension (512 is CLIP ViT-B/32)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="rows per timed add"
    )
    parser.add_argument(
        "--repeats", type=int, default=20, help="timed operations per size"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="neighbors per query"
    )
    parser.add_argument("--seed", type=int, default=51, help="random seed")
    parser.add_argument(
        "--tail",
        action="store_true",
        help="sweep add cost against a growing unindexed tail instead of "
        "against table size",
    )
    parser.add_argument(
        "--grow-batch",
        type=int,
        default=_TAIL_GROW_BATCH,
        help="rows per bulk add while growing the tail, with --tail. Every "
        "add appends a fragment, so this separates tail cost from fragment "
        "count: the same tail reached in few large adds or many small ones",
    )
    parser.add_argument(
        "--checkpoints",
        default="0,1000,5000,20000,50000,100000,200000,500000",
        help="tail sizes to measure at, with --tail",
    )
    args = parser.parse_args()

    sizes = [int(size) for size in args.sizes.split(",")]

    # The tail sweep times writes only, which need no sample collection and
    # so no database
    dataset = None
    if not args.tail:
        dataset = fo.Dataset(_DATASET_NAME, overwrite=True)
        dataset.add_samples(
            [
                fo.Sample(filepath=os.path.join(tempfile.gettempdir(), name))
                for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")
            ]
        )

    print(
        "LanceDB %s | dims=%d batch=%d repeats=%d k=%d"
        % (
            lancedb.__version__,
            args.dims,
            args.batch_size,
            args.repeats,
            args.k,
        )
    )

    if args.tail:
        _run_tail_sweep(
            dataset,
            sizes[0],
            dims=args.dims,
            batch_size=args.batch_size,
            repeats=args.repeats,
            seed=args.seed,
            checkpoints=[int(c) for c in args.checkpoints.split(",")],
            grow_batch=args.grow_batch,
        )
        return

    results = []
    try:
        for num_rows in sizes:
            print("\n%d rows" % num_rows)
            results.append(
                (num_rows,)
                + _run_size(
                    dataset,
                    num_rows,
                    dims=args.dims,
                    batch_size=args.batch_size,
                    repeats=args.repeats,
                    k=args.k,
                    seed=args.seed,
                )
            )
    finally:
        dataset.delete()

    print(
        "\n%-12s %14s %16s %16s"
        % ("rows", "add (ms)", "query (ms)", "remove (ms)")
    )
    for num_rows, add_ms, query_ms, remove_ms in results:
        print(
            "%-12d %14.1f %16.1f %16.1f"
            % (num_rows, add_ms, query_ms, remove_ms)
        )


if __name__ == "__main__":
    main()
