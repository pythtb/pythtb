"""Tests for the npz cache in the W90 loader."""

import shutil
from pathlib import Path

import numpy as np

from pythtb.io.w90 import load_w90_dataset

FIXTURE = Path(__file__).resolve().parents[1] / "w90_berry" / "hBN"
PREFIX = "hBN"


def _copy_fixture(tmp_path):
    for f in FIXTURE.iterdir():
        if f.is_file() and not f.name.endswith("_pythtb_cache.npz"):
            shutil.copy(f, tmp_path / f.name)
    return tmp_path


def _datasets_equal(a, b):
    assert a.num_wan == b.num_wan
    assert set(a.ham_r) == set(b.ham_r)
    for R in a.ham_r:
        np.testing.assert_array_equal(a.ham_r[R].h, b.ham_r[R].h)
        assert a.ham_r[R].degeneracy == b.ham_r[R].degeneracy
    assert (a.pos_r is None) == (b.pos_r is None)
    if a.pos_r is not None:
        assert set(a.pos_r) == set(b.pos_r)
        for R in a.pos_r:
            np.testing.assert_array_equal(a.pos_r[R], b.pos_r[R])


def test_cache_roundtrip(tmp_path):
    root = _copy_fixture(tmp_path)
    cache_file = root / f"{PREFIX}_pythtb_cache.npz"

    fresh = load_w90_dataset(root, PREFIX, cache=False)
    assert not cache_file.exists()

    first = load_w90_dataset(root, PREFIX)  # parses and writes the cache
    assert cache_file.exists()
    cached = load_w90_dataset(root, PREFIX)  # served from the cache

    _datasets_equal(fresh, first)
    _datasets_equal(fresh, cached)


def test_cache_invalidated_by_source_change(tmp_path):
    root = _copy_fixture(tmp_path)
    cache_file = root / f"{PREFIX}_pythtb_cache.npz"
    load_w90_dataset(root, PREFIX)
    assert cache_file.exists()

    # Touch a source file: signature changes, the stale cache is ignored and
    # rewritten, and the data still round-trips.
    src = root / f"{PREFIX}_tb.dat"
    stat = src.stat()
    import os

    os.utime(src, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    refreshed = load_w90_dataset(root, PREFIX)
    fresh = load_w90_dataset(root, PREFIX, cache=False)
    _datasets_equal(fresh, refreshed)


def test_corrupt_cache_falls_back(tmp_path):
    root = _copy_fixture(tmp_path)
    cache_file = root / f"{PREFIX}_pythtb_cache.npz"
    cache_file.write_bytes(b"not an npz file")
    ds = load_w90_dataset(root, PREFIX)
    fresh = load_w90_dataset(root, PREFIX, cache=False)
    _datasets_equal(fresh, ds)
