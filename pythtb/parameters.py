"""Parameter registration, partitioning, and sweep evaluation for TBModel.

This module owns the three concepts behind parameterized tight-binding terms:

1. :class:`ParamTerm` — one normalized provider. Strings are compiled once at
   registration and their free parameter names extracted by parsing (so
   ``"t1"``, ``"-t1"``, and ``"t1*np.exp(1j*phi)"`` all work and report the
   names they actually need); callables have their signatures resolved once.
2. :class:`ParameterRegistry` — the set of parameterized on-site and hopping
   terms registered on a model, with name bookkeeping (``names``, ``missing``)
   and display helpers.
3. :class:`SweepSpec` — the partition of the ``**params`` of a single call
   into scalar assignments and cartesian sweep axes, the grid evaluator, and
   the finite-difference metadata for swept axes.

The shape rules for a parameter value are strict and explicit:

========================  =========================================
value                     meaning
========================  =========================================
scalar / 0-D array        single value
1-D list/tuple/array      sweep axis (cartesian product with others)
(1, n) 2-D array          single vector value
(2, 2) 2-D array          single matrix value (spinful models)
(n, 4) 2-D array          sweep over Pauli 4-vectors (spinful models)
anything else             ``ValueError``
========================  =========================================
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Mapping

import numpy as np

__all__ = []  # internal machinery; nothing exported into the pythtb namespace

# Names available inside string-expression providers (besides parameters).
_EXPR_ENV = {"np": np, "numpy": np, "pi": np.pi, "complex": complex, "float": float}


def _expr_free_names(expr: str) -> tuple[str, ...]:
    """Free parameter names of a string expression (parsed, not guessed)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Parameter expression {expr!r} is not a valid Python expression: {exc}"
        ) from None
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    free = names - set(_EXPR_ENV)
    if not free:
        raise ValueError(
            f"Parameter expression {expr!r} contains no free parameter names."
        )
    return tuple(sorted(free))


@lru_cache(maxsize=None)
def _signature_info(f: Callable) -> tuple[bool, tuple[str, ...]]:
    """(accepts **kwargs, required keyword names) of a callable, resolved once."""
    sig = inspect.signature(f)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    required = tuple(
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.default is inspect._empty
    )
    return accepts_kwargs, required


@dataclass
class ParamTerm:
    """One parameterized on-site or hopping provider, normalized at registration."""

    provider: object  # original string or callable (kept for display/copy)
    names: tuple[str, ...]  # free parameter names this term consumes
    accepts_any: bool  # callable with **kwargs takes the full assignment dict
    _func: Callable = field(repr=False)

    @classmethod
    def from_provider(cls, provider, *, ctx: str) -> "ParamTerm":
        """Normalize a string expression or callable into a :class:`ParamTerm`."""
        if isinstance(provider, str):
            names = _expr_free_names(provider)
            code = compile(provider, f"<pythtb parameter {ctx}>", "eval")

            def func(assignments, _code=code):
                env = dict(_EXPR_ENV)
                env.update(assignments)
                return eval(_code, {"__builtins__": {}}, env)

            return cls(provider=provider, names=names, accepts_any=False, _func=func)

        if callable(provider):
            accepts_kwargs, required = _signature_info(provider)
            if not required:
                raise ValueError(f"{ctx} callable must declare at least one parameter.")
            if accepts_kwargs:
                return cls(
                    provider=provider,
                    names=tuple(required),
                    accepts_any=True,
                    _func=lambda assignments, _p=provider: _p(**assignments),
                )

            def func(assignments, _p=provider, _req=required):
                return _p(**{k: assignments[k] for k in _req if k in assignments})

            return cls(
                provider=provider, names=tuple(required), accepts_any=False, _func=func
            )

        raise TypeError(f"Unsupported {ctx} provider type: {type(provider)}")

    def evaluate(self, assignments: Mapping[str, object]):
        """Value of this term under the given parameter assignments."""
        try:
            return self._func(assignments)
        except NameError as exc:
            missing = exc.args[0].split("'")[1]
            raise ValueError(
                f"Expression {self.provider!r} needs a value for parameter '{missing}'."
            ) from None
        except KeyError as exc:
            raise ValueError(
                f"Provider {self.describe()} needs a value for parameter {exc}."
            ) from None

    def covered_by(self, given: Iterable[str]) -> bool:
        """True when the given names fully determine this term."""
        return set(self.names) <= set(given)

    def describe(self) -> str:
        """Human-readable description of the provider for info displays."""
        if isinstance(self.provider, str):
            return f"'{self.provider}'"
        try:
            return inspect.getsource(self.provider).strip()
        except (OSError, TypeError, AttributeError):
            name = getattr(
                self.provider,
                "__qualname__",
                getattr(self.provider, "__name__", repr(self.provider)),
            )
            return f"callable {name} (source unavailable)"


class ParameterRegistry:
    """Parameterized on-site and hopping terms registered on one model."""

    def __init__(self):
        """Create an empty registry."""
        self.onsite: dict[int, ParamTerm] = {}
        self.hoppings: dict[tuple, ParamTerm] = {}

    def __bool__(self) -> bool:
        """True when any parameterized term is registered."""
        return bool(self.onsite) or bool(self.hoppings)

    def register_onsite(self, idx: int, provider) -> ParamTerm:
        """Normalize and store a provider for the on-site term at ``idx``."""
        term = ParamTerm.from_provider(provider, ctx=f"onsite[{idx}]")
        self.onsite[idx] = term
        return term

    def register_hopping(self, key: tuple, provider) -> ParamTerm:
        """Normalize and store a provider for the hopping at ``key = (i, j, R)``."""
        i, j, R = key[0], key[1], tuple(key[2])
        term = ParamTerm.from_provider(provider, ctx=f"hopping[{i},{j},{R}]")
        self.hoppings[key] = term
        return term

    def discard(self, *, onsite_idx=None, hop_key=None) -> None:
        """Remove the provider at an on-site index and/or hopping key, if present."""
        if onsite_idx is not None:
            self.onsite.pop(onsite_idx, None)
        if hop_key is not None:
            self.hoppings.pop(hop_key, None)

    @property
    def names(self) -> tuple[str, ...]:
        """Sorted union of all free parameter names across registered terms."""
        out: set[str] = set()
        for term in self.onsite.values():
            out.update(term.names)
        for term in self.hoppings.values():
            out.update(term.names)
        return tuple(sorted(out))

    def missing(self, given: Iterable[str]) -> tuple[str, ...]:
        """Registered parameter names not covered by the given assignment names."""
        return tuple(n for n in self.names if n not in set(given))

    def copy(self) -> "ParameterRegistry":
        """Shallow copy sharing the (immutable) term objects."""
        new = ParameterRegistry()
        new.onsite = dict(self.onsite)
        new.hoppings = dict(self.hoppings)
        return new


@dataclass(frozen=True)
class AxisFD:
    """Finite-difference metadata for one swept parameter axis."""

    name: str
    index: int  # position among the sweep axes
    step: float
    periodic: bool
    trimmed: bool  # endpoint duplicated the start and was dropped


def normalize_axis(values, *, name: str, period: float | None = None):
    """Normalize a 1D parameter sweep and report finite-difference metadata.

    Returns
    -------
    values_unique : np.ndarray
        Copy of the input with any duplicated endpoint removed.
    step : float
        Uniform spacing between samples (computed before trimming).
    is_periodic : bool
        True if the sweep spans a full cycle.
    trimmed : bool
        True when the final element duplicated the first and was dropped.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(
            f"Parameter '{name}' must be one-dimensional with at least two samples."
        )

    diffs = np.diff(arr)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(f"Parameter '{name}' must be uniformly spaced.")
    step = float(diffs[0])

    periodic = False
    trimmed = False
    if period is not None:
        period = float(period)
        span = arr[-1] - arr[0]
        if np.isclose(span, period):
            arr = arr[:-1]
            periodic = True
            trimmed = True
        elif np.isclose(step * arr.size, period):
            periodic = True
    else:
        if np.isclose(arr[-1], arr[0]):
            arr = arr[:-1]
            periodic = True
            trimmed = True

    return arr.copy(), step, periodic, trimmed


@dataclass(frozen=True)
class SweepSpec:
    """Scalar assignments plus cartesian sweep axes for one method call."""

    scalars: dict[str, object]
    names: tuple[str, ...]
    axes: tuple[tuple, ...]

    @classmethod
    def from_params(cls, params: Mapping[str, object], *, spinful: bool) -> "SweepSpec":
        """Partition call parameters by the documented shape rules (strict)."""
        scalars: dict[str, object] = {}
        sweep_names: list[str] = []
        sweep_axes: list[tuple] = []

        def reject(name, raw, why):
            raise ValueError(
                f"Parameter '{name}' has unsupported {why}. Accepted forms: "
                "scalar, 1-D array (sweep), (1, n) vector value, "
                "(2, 2) matrix value (spinful), (n, 4) Pauli-vector sweep (spinful)."
            )

        for name, raw in params.items():
            if isinstance(raw, (int, float, complex)) and not isinstance(raw, bool):
                scalars[name] = raw
                continue
            if not isinstance(raw, (list, tuple, np.ndarray)):
                reject(name, raw, f"type {type(raw).__name__}")

            arr = np.asarray(raw)
            if arr.ndim == 0:
                scalars[name] = arr.item()
            elif arr.ndim == 1:
                if arr.shape[0] == 0:
                    raise ValueError(
                        f"Parameter sweep '{name}' must provide at least one value."
                    )
                # keep original elements so providers see the user's dtype
                sweep_names.append(name)
                sweep_axes.append(tuple(arr[i] for i in range(arr.shape[0])))
            elif arr.ndim == 2 and arr.shape[0] == 1:
                scalars[name] = arr[0, :].copy()  # single vector value
            elif arr.ndim == 2 and arr.shape == (2, 2):
                if not spinful:
                    raise ValueError(
                        f"Parameter '{name}' is a 2x2 array, but the model is spinless."
                    )
                scalars[name] = arr.copy()
            elif arr.ndim == 2 and arr.shape[1] == 4:
                if not spinful:
                    raise ValueError(
                        f"Parameter '{name}' has shape {arr.shape}, but the model "
                        "is spinless."
                    )
                sweep_names.append(name)
                sweep_axes.append(tuple(arr[i, :].copy() for i in range(arr.shape[0])))
            else:
                reject(name, raw, f"shape {arr.shape}")

        return cls(scalars=scalars, names=tuple(sweep_names), axes=tuple(sweep_axes))

    @property
    def has_axes(self) -> bool:
        """True when at least one parameter is swept over an axis."""
        return bool(self.axes)

    def assignment_names(self) -> tuple[str, ...]:
        """All parameter names this call assigns (scalars first, then sweeps)."""
        return tuple(self.scalars) + self.names

    def fd_axes(
        self, param_periods: Mapping[str, float] | None
    ) -> tuple[list[np.ndarray], list[AxisFD]]:
        """Float evaluation axes plus finite-difference metadata per sweep axis.

        Axes that are not differentiable (fewer than two samples, or
        non-numeric) are returned in ``raw_axes`` but get no ``AxisFD`` entry.
        """
        periods = dict(param_periods or {})
        raw_axes: list[np.ndarray] = []
        specs: list[AxisFD] = []
        for idx, name in enumerate(self.names):
            axis_array = np.asarray(self.axes[idx], dtype=float)
            raw_axes.append(axis_array.copy())
            if axis_array.ndim != 1 or axis_array.size < 2:
                continue
            _, step, periodic, trimmed = normalize_axis(
                axis_array, name=name, period=periods.get(name)
            )
            specs.append(
                AxisFD(
                    name=name, index=idx, step=step, periodic=periodic, trimmed=trimmed
                )
            )
        return raw_axes, specs

    def evaluate(self, build_fn, n_lead: tuple[int, ...], *, axes=None):
        """Evaluate ``build_fn`` once or over the cartesian product of axes.

        ``build_fn(assignments)`` must return a tuple of arrays whose shapes do
        not depend on the assignment. Sweep axes are inserted after the first
        ``n_lead[i]`` dimensions of each returned array.
        """
        eval_axes = self.axes if axes is None else tuple(axes)
        if len(eval_axes) != len(self.names):
            raise ValueError("Number of parameter sweep axes must match names.")
        if not eval_axes:
            return tuple(build_fn(self.scalars))

        axis_lengths = [len(axis) for axis in eval_axes]
        n_blocks = int(np.prod(axis_lengths, dtype=int))
        stacked_flat = None
        assignments = self.scalars.copy()

        for flat_idx, multi_idx in enumerate(np.ndindex(*axis_lengths)):
            for axis_idx, name in enumerate(self.names):
                assignments[name] = eval_axes[axis_idx][multi_idx[axis_idx]]

            blocks = tuple(build_fn(assignments))
            if stacked_flat is None:
                stacked_flat = [
                    np.empty((n_blocks, *block.shape), dtype=block.dtype)
                    for block in blocks
                ]
            for out, block in zip(stacked_flat, blocks, strict=True):
                out[flat_idx] = block

        p = len(axis_lengths)
        results = []
        for out, lead in zip(stacked_flat, n_lead, strict=True):
            out = out.reshape(*axis_lengths, *out.shape[1:])
            results.append(np.moveaxis(out, range(p), range(lead, lead + p)))
        return tuple(results)
