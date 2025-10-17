from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class HoppingTable:
    """Array-backed storage with fast mutation/query helpers for hoppings."""

    dim_r: int
    spinful: bool
    amplitudes: np.ndarray = field(init=False)
    from_idx: np.ndarray = field(init=False)
    to_idx: np.ndarray = field(init=False)
    lattice_vecs: np.ndarray = field(init=False)
    _index: dict[tuple[int, int, tuple[int, ...]], int] = field(init=False, default_factory=dict)
    _flatten_cache: dict[tuple[int, int], dict[str, np.ndarray]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        self.clear()

    # ------------------------------------------------------------------
    # core protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.from_idx.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield (
                self.amplitudes[idx],
                int(self.from_idx[idx]),
                int(self.to_idx[idx]),
                self.lattice_vecs[idx].copy(),
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        if self.spinful:
            self.amplitudes = np.empty((0, 2, 2), dtype=complex)
        else:
            self.amplitudes = np.empty((0,), dtype=complex)
        self.from_idx = np.empty((0,), dtype=int)
        self.to_idx = np.empty((0,), dtype=int)
        self.lattice_vecs = np.empty((0, self.dim_r), dtype=int)
        self._index.clear()
        self._flatten_cache.clear()

    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.amplitudes, self.from_idx, self.to_idx, self.lattice_vecs

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def append(self, amplitude: np.ndarray, i: int, j: int, R: Sequence[int]) -> int:
        amp = self._coerce_amplitude(amplitude)
        R_arr = np.asarray(R, dtype=int).reshape(self.dim_r)

        if self.spinful:
            self.amplitudes = (
                np.concatenate([self.amplitudes, amp[np.newaxis, ...]]) if len(self) else amp[np.newaxis, ...]
            )
        else:
            scalar = np.asarray(amp, dtype=complex).reshape(())
            self.amplitudes = (
                np.concatenate([self.amplitudes, [scalar]])
                if len(self)
                else np.asarray([scalar], dtype=complex)
            )

        self.from_idx = np.append(self.from_idx, int(i))
        self.to_idx = np.append(self.to_idx, int(j))
        self.lattice_vecs = (
            np.vstack([self.lattice_vecs, R_arr])
            if self.lattice_vecs.size
            else R_arr[np.newaxis, :]
        )

        idx = len(self) - 1
        # store mapping (i, j, R) -> index for O(1) lookup later
        self._index[self._make_key(i, j, R_arr)] = idx
        self._flatten_cache.clear()
        return idx

    def extend(
        self,
        amplitudes: Sequence[np.ndarray],
        i_idx: Sequence[int],
        j_idx: Sequence[int],
        lattice_vecs: Sequence[Sequence[int]],
    ) -> None:
        if not amplitudes:
            return

        # Coerce indices/lattice vectors into contiguous arrays and validate lengths
        i_arr = np.asarray(list(i_idx), dtype=int)
        j_arr = np.asarray(list(j_idx), dtype=int)
        R_arr = np.asarray(list(lattice_vecs), dtype=int).reshape(len(i_arr), self.dim_r)

        if i_arr.size == 0:
            return
        if not (len(amplitudes) == i_arr.size == j_arr.size == R_arr.shape[0]):
            raise ValueError("Lengths of amplitudes, i_idx, j_idx, and lattice_vecs must match.")

        start = len(self)
        if self.spinful:
            amp_list = [self._coerce_amplitude(amp) for amp in amplitudes]
            new_amps = np.stack(amp_list, axis=0)
            self.amplitudes = (
                np.concatenate([self.amplitudes, new_amps]) if start else new_amps
            )
        else:
            amp_array = np.asarray(
                [self._coerce_amplitude(amp) for amp in amplitudes], dtype=complex
            ).reshape(-1)
            self.amplitudes = (
                np.concatenate([self.amplitudes, amp_array]) if start else amp_array
            )

        self.from_idx = (
            np.concatenate([self.from_idx, i_arr]) if start else i_arr.copy()
        )
        self.to_idx = (
            np.concatenate([self.to_idx, j_arr]) if start else j_arr.copy()
        )
        self.lattice_vecs = (
            np.vstack([self.lattice_vecs, R_arr]) if start else R_arr.copy()
        )

        # freshly inserted entries start at `start`; update lookup index
        for offset, (i, j, R_vec) in enumerate(zip(i_arr, j_arr, R_arr, strict=True)):
            key = self._make_key(i, j, R_vec)
            self._index[key] = start + offset

        self._flatten_cache.clear()

    def update(self, idx: int, *, amplitude=None, R=None) -> None:
        old_key = self._make_key(self.from_idx[idx], self.to_idx[idx], self.lattice_vecs[idx])
        if amplitude is not None:
            amp = self._coerce_amplitude(amplitude)
            if self.spinful:
                self.amplitudes[idx] = amp
            else:
                self.amplitudes[idx] = np.asarray(amp, dtype=complex).reshape(())
        if R is not None:
            self.lattice_vecs[idx] = np.asarray(R, dtype=int).reshape(self.dim_r)
        new_key = self._make_key(self.from_idx[idx], self.to_idx[idx], self.lattice_vecs[idx])
        if new_key != old_key:
            self._index.pop(old_key, None)
            self._index[new_key] = idx
        self._flatten_cache.clear()

    def accumulate(self, idx: int, delta: np.ndarray) -> None:
        if self.spinful:
            self.amplitudes[idx] += np.asarray(delta, dtype=complex)
        else:
            self.amplitudes[idx] += np.asarray(delta, dtype=complex).reshape(())
        self._flatten_cache.clear()

    def remove_orbitals(self, indices: Sequence[int]) -> None:
        unique = sorted(set(int(i) for i in indices))
        if not unique or len(self) == 0:
            return

        keep_mask = np.ones(len(self), dtype=bool)
        for orb in unique:
            keep_mask &= (self.from_idx != orb) & (self.to_idx != orb)

        self._apply_mask(keep_mask)

        for orb in unique:
            self.from_idx[self.from_idx > orb] -= 1
            self.to_idx[self.to_idx > orb] -= 1

        self._rebuild_index()
        self._flatten_cache.clear()

    def shift_orbital(self, orb_idx: int, disp_vec: Sequence[int]) -> None:
        if len(self) == 0:
            return
        disp = np.asarray(disp_vec, dtype=int).reshape(self.dim_r)
        if not np.any(disp):
            return

        mask_from = self.from_idx == orb_idx
        mask_to = self.to_idx == orb_idx
        if np.any(mask_from):
            self.lattice_vecs[mask_from] -= disp
        if np.any(mask_to):
            self.lattice_vecs[mask_to] += disp

        if np.any(mask_from) or np.any(mask_to):
            self._rebuild_index()
            self._flatten_cache.clear()

    def normalize_entry(
        self,
        ind_i,
        ind_j,
        ind_R,
        *,
        norb: int,
        dim_k: int,
        periodic_dirs: Sequence[int],
    ) -> tuple[int, int, np.ndarray]:
        """Validate hopping indices and lattice vector, returning canonical forms."""
        if not isinstance(ind_i, (int, np.integer)):
            raise TypeError("Index ind_i is not within range of number of orbitals.")
        if not isinstance(ind_j, (int, np.integer)):
            raise TypeError("Index ind_j is not within range of number of orbitals.")

        ind_i = int(ind_i)
        ind_j = int(ind_j)

        if ind_i < 0 or ind_i >= norb:
            raise ValueError("Index ind_i is not within range of number of orbitals.")
        if ind_j < 0 or ind_j >= norb:
            raise ValueError("Index ind_j is not within range of number of orbitals.")

        if dim_k == 0:
            if ind_R is not None:
                raise ValueError("No periodic directions, so ind_R should not be specified.")
            return ind_i, ind_j, np.zeros(self.dim_r, dtype=int)

        if ind_R is None:
            raise ValueError("Must specify ind_R when we have a periodic direction.")

        if isinstance(ind_R, np.ndarray):
            if ind_R.ndim != 1:
                raise ValueError("If ind_R is a numpy array, it must be 1-dimensional.")
            if ind_R.shape[0] != self.dim_r:
                raise ValueError("If ind_R is a numpy array, its length must equal dim_r.")
            if not np.issubdtype(ind_R.dtype, np.integer):
                raise ValueError("If ind_R is a numpy array, it must be of integer type.")
            R_vec = ind_R.astype(int, copy=False)
        elif isinstance(ind_R, (list, tuple)):
            if len(ind_R) != self.dim_r:
                raise ValueError("If ind_R is a list or tuple, its length must equal dim_r.")
            R_vec = np.asarray(ind_R, dtype=int)
        elif isinstance(ind_R, (int, np.integer)):
            if dim_k != 1:
                raise ValueError("If dim_k is not 1, should not use integer for ind_R. Instead use list.")
            R_vec = np.zeros(self.dim_r, dtype=int)
            R_vec[list(periodic_dirs)] = int(ind_R)
        else:
            raise TypeError("ind_R is not of correct type. Should be array-type or integer.")

        return ind_i, ind_j, R_vec.reshape(self.dim_r)

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    def find(self, i: int, j: int, R: Sequence[int]) -> int | None:
        return self._index.get(self._make_key(i, j, R))

    # ------------------------------------------------------------------
    # cached utilities
    # ------------------------------------------------------------------
    def flatten_cache(self, norb: int) -> dict[str, np.ndarray]:
        key = (norb, len(self))
        cache = self._flatten_cache.get(key)
        if cache is not None:
            return cache

        i_indices = self.from_idx
        j_indices = self.to_idx

        if len(self) == 0:
            cache = {
                "order": np.empty((0,), dtype=int),
                "starts": np.empty((0,), dtype=int),
                "uniq": np.empty((0,), dtype=int),
                "cols_transposed": np.empty((0,), dtype=int),
                "inverse_order": np.empty((0,), dtype=int),
            }
            self._flatten_cache[key] = cache
            return cache

        flat_idx = i_indices * norb + j_indices
        order = np.argsort(flat_idx, kind="mergesort")
        flat_sorted = flat_idx[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(flat_sorted)) + 1))
        uniq = flat_sorted[starts]

        rows = uniq // norb
        cols = uniq % norb
        cols_transposed = cols * norb + rows

        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size)

        cache = {
            "order": order,
            "starts": starts,
            "uniq": uniq,
            "cols_transposed": cols_transposed,
            "inverse_order": inverse_order,
        }
        self._flatten_cache[key] = cache
        return cache

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _coerce_amplitude(self, amp):
        if self.spinful:
            arr = np.asarray(amp, dtype=complex)
            if arr.shape != (2, 2):
                raise ValueError("Spinful hopping amplitudes must be 2x2 matrices.")
            return arr
        return np.asarray(amp, dtype=complex).reshape(())

    def _make_key(self, i: int, j: int, R: Sequence[int]) -> tuple[int, int, tuple[int, ...]]:
        R_arr = np.asarray(R, dtype=int).reshape(self.dim_r)
        return (int(i), int(j), tuple(int(x) for x in R_arr))

    def _apply_mask(self, mask: np.ndarray) -> None:
        self.amplitudes = self.amplitudes[mask]
        self.from_idx = self.from_idx[mask]
        self.to_idx = self.to_idx[mask]
        self.lattice_vecs = self.lattice_vecs[mask]

    def _rebuild_index(self) -> None:
        self._index.clear()
        for idx in range(len(self)):
            key = self._make_key(self.from_idx[idx], self.to_idx[idx], self.lattice_vecs[idx])
            self._index[key] = idx
