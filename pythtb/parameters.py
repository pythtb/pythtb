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

# ---------------------------------------------------------------------------
# MENTAL MODEL (read this first)
# ---------------------------------------------------------------------------
# A "parameter" is a symbol whose value is supplied later, not when you build
# the model. There are two moments in time:
#
#   (1) MODEL-BUILD TIME — when you call tb.set_onsite("U", ...) or
#       tb.set_hop("t1*np.exp(1j*phi)", ...). At this moment we DON'T know the
#       numeric value of U / t1 / phi. We just record HOW to compute the term
#       once values arrive. That recorded recipe is a `ParamTerm`, and the bag
#       of all such recipes for a model is a `ParameterRegistry`.
#
#   (2) CALL TIME — when you call tb.hamiltonian(k, U=0.3, phi=[0, 1, 2]).
#       Now concrete values arrive as **params. A `SweepSpec` sorts those values
#       into "scalars" (one fixed value) and "sweep axes" (a list of values to
#       loop over), then drives evaluation, building one Hamiltonian per
#       combination and stacking the results into a single array.
#
# So: ParamTerm/Registry = "what the symbols mean" (set once at build time).
#     SweepSpec/AxisFD    = "what values to plug in this call" (per call).
# AxisFD additionally carries the spacing/periodicity a swept axis needs so the
# velocity code can finite-difference dH/d(parameter) along it.
# ---------------------------------------------------------------------------

# Names a string-expression provider may use BESIDES its free parameters.
# When we scan "t1*np.exp(1j*phi)" for parameter names, anything in this dict
# (np, numpy, pi, complex, float) is treated as "already known", not a free
# parameter the user must supply. This same dict is the evaluation environment.
_EXPR_ENV = {"np": np, "numpy": np, "pi": np.pi, "complex": complex, "float": float}


def _expr_free_names(expr: str) -> tuple[str, ...]:
    """Free parameter names of a string expression (parsed, not guessed).

    We parse the expression into an Abstract Syntax Tree and collect every bare
    NAME token, then subtract the always-available names in ``_EXPR_ENV``. What
    remains are the symbols the user must provide values for.

    Worked example: ``"t1*np.exp(1j*phi)"``
      - AST Name nodes found  -> {"t1", "np", "phi"}
        (``exp`` is an *attribute* of ``np``, not a bare Name; ``1j`` is a
        numeric constant — neither shows up as a Name.)
      - subtract _EXPR_ENV     -> drop "np"
      - free names (sorted)    -> ("phi", "t1")
    """
    try:
        # mode="eval" => parse a single expression (not statements).
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        # e.g. "t1 +" or "import os" — not a usable scalar expression.
        raise ValueError(
            f"Parameter expression {expr!r} is not a valid Python expression: {exc}"
        ) from None
    # ast.walk yields every node; keep the identifier tokens (ast.Name) only.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    free = names - set(_EXPR_ENV)  # remove np/pi/complex/... leaving real params
    if not free:
        # A "parameter" with no symbols (e.g. "2*np.pi") is almost surely a
        # mistake — the user meant to pass a number, not register a parameter.
        raise ValueError(
            f"Parameter expression {expr!r} contains no free parameter names."
        )
    return tuple(sorted(free))


@lru_cache(maxsize=None)  # a given callable's signature never changes; cache it
def _signature_info(f: Callable) -> tuple[bool, tuple[str, ...]]:
    """(accepts **kwargs, required keyword names) of a callable, resolved once.

    We introspect a user-supplied callable provider so we know (a) whether it
    can swallow the whole assignment dict via ``**kwargs`` and (b) which named
    arguments it *requires* (those without a default). Examples:

      lambda t, phi: ...        -> (False, ("t", "phi"))
      def f(t, **extra): ...    -> (True,  ("t",))
      def f(**kw): ...          -> (True,  ())
    """
    sig = inspect.signature(f)
    # Does the signature contain a **kwargs catch-all?
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    # Required = ordinary/keyword-only params that have no default value.
    required = tuple(
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.default is inspect._empty  # inspect._empty == "no default given"
    )
    return accepts_kwargs, required


# A @dataclass auto-generates __init__/__repr__/__eq__ from the fields below,
# so we don't have to hand-write them.
@dataclass
class ParamTerm:
    """One parameterized on-site or hopping provider, normalized at registration.

    Think of this as a frozen "recipe": given a dict of parameter values, it can
    produce the numeric on-site energy or hopping amplitude. The two user-facing
    provider kinds (a string expression or a Python callable) are both reduced to
    a single uniform interface here: ``term.evaluate(assignments) -> value``.
    """

    provider: object  # ORIGINAL string or callable, kept verbatim for display/copy
    names: tuple[str, ...]  # the free parameter names this term needs values for
    accepts_any: bool  # True iff a callable provider has **kwargs (takes everything)
    # The compiled "do the work" function. field(repr=False) hides it from the
    # auto-generated __repr__ (a compiled closure prints as ugly noise).
    _func: Callable = field(repr=False)

    @classmethod
    def from_provider(cls, provider, *, ctx: str) -> "ParamTerm":
        """Normalize a string expression or callable into a :class:`ParamTerm`.

        ``ctx`` is just a label (e.g. ``"onsite[3]"`` or ``"hopping[0,1,(0,0)]"``)
        used in error messages and the compiled code's filename, so tracebacks
        point back to the term that caused them. Called ONCE per term, at
        registration — the expensive parsing/compiling/introspection happens here,
        not on every Hamiltonian build.
        """
        # ---- Case 1: a string expression, e.g. "t1*0.5" or "delta*np.cos(phi)"
        if isinstance(provider, str):
            names = _expr_free_names(provider)  # which symbols must be supplied
            # Compile once now; the resulting code object is re-eval'd cheaply later.
            code = compile(provider, f"<pythtb parameter {ctx}>", "eval")

            def func(assignments, _code=code):
                # Build the evaluation namespace: known names (np, pi, ...) plus
                # the user's parameter values. `assignments` wins on name clashes.
                env = dict(_EXPR_ENV)
                env.update(assignments)
                # Sandbox: globals has NO __builtins__, so the expression can't
                # call open()/import/etc. `env` is the locals it may reference.
                return eval(_code, {"__builtins__": {}}, env)

            return cls(provider=provider, names=names, accepts_any=False, _func=func)

        # ---- Case 2: a Python callable, e.g. lambda t, phi: t*np.exp(1j*phi)
        if callable(provider):
            accepts_kwargs, required = _signature_info(provider)
            if not required:
                # A 0-argument callable isn't parameterized by anything — reject,
                # the user should just pass the value it would have returned.
                raise ValueError(f"{ctx} callable must declare at least one parameter.")
            if accepts_kwargs:
                # Has **kwargs => safe to forward the WHOLE assignment dict; it
                # will pick out what it wants and ignore the rest.
                return cls(
                    provider=provider,
                    names=tuple(required),
                    accepts_any=True,
                    _func=lambda assignments, _p=provider: _p(**assignments),
                )

            # No **kwargs => only hand it the exact names it declared, otherwise
            # Python would raise "unexpected keyword argument" for extras.
            def func(assignments, _p=provider, _req=required):
                return _p(**{k: assignments[k] for k in _req if k in assignments})

            return cls(
                provider=provider, names=tuple(required), accepts_any=False, _func=func
            )

        # ---- Anything else (int passed as a "parameter", etc.) is a usage error.
        raise TypeError(f"Unsupported {ctx} provider type: {type(provider)}")

    def evaluate(self, assignments: Mapping[str, object]):
        """Value of this term under the given parameter assignments.

        The try/except turns the two "you forgot to supply a value" failure modes
        into one clear, actionable error message:
          - string path raises NameError  (a symbol in the expr was undefined)
          - callable path raises KeyError (a required kwarg wasn't in assignments)
        """
        try:
            return self._func(assignments)
        except NameError as exc:
            # exc message looks like: "name 'phi' is not defined" — pull out 'phi'.
            missing = exc.args[0].split("'")[1]
            raise ValueError(
                f"Expression {self.provider!r} needs a value for parameter '{missing}'."
            ) from None
        except KeyError as exc:
            # exc is the missing key itself (already quoted when str-formatted).
            raise ValueError(
                f"Provider {self.describe()} needs a value for parameter {exc}."
            ) from None

    def covered_by(self, given: Iterable[str]) -> bool:
        """True when the given names fully determine this term.

        i.e. every name this term needs is present in ``given`` (subset test).
        """
        return set(self.names) <= set(given)

    def describe(self) -> str:
        """Human-readable description of the provider for info displays."""
        # Strings describe themselves.
        if isinstance(self.provider, str):
            return f"'{self.provider}'"
        try:
            # For a callable, show its source line(s) if we can recover them
            # (works for `def`/`lambda` defined in a file, not in the REPL).
            return inspect.getsource(self.provider).strip()
        except (OSError, TypeError, AttributeError):
            # Source not available (REPL, built-in, C function): fall back to a
            # name. Prefer the fully-qualified name, then __name__, then repr().
            name = getattr(
                self.provider,
                "__qualname__",
                getattr(self.provider, "__name__", repr(self.provider)),
            )
            return f"callable {name} (source unavailable)"


class ParameterRegistry:
    """Parameterized on-site and hopping terms registered on one model.

    Just two dictionaries of :class:`ParamTerm` recipes — one keyed by orbital
    index (on-site energies), one keyed by hopping identity ``(i, j, R)`` — plus
    convenience queries over the union of parameter names they reference. Each
    TBModel owns exactly one of these.
    """

    def __init__(self):
        """Create an empty registry."""
        # orbital index            -> recipe for that site's on-site energy
        self.onsite: dict[int, ParamTerm] = {}
        # (i, j, R) hopping key     -> recipe for that hopping amplitude
        self.hoppings: dict[tuple, ParamTerm] = {}

    def __bool__(self) -> bool:
        """True when any parameterized term is registered.

        Lets callers write ``if registry:`` / ``if not registry:`` to ask
        "does this model have any parameters at all?".
        """
        return bool(self.onsite) or bool(self.hoppings)

    def register_onsite(self, idx: int, provider) -> ParamTerm:
        """Normalize and store a provider for the on-site term at ``idx``."""
        # Build the recipe (parses/compiles now) and remember it under this index.
        term = ParamTerm.from_provider(provider, ctx=f"onsite[{idx}]")
        self.onsite[idx] = term
        return term

    def register_hopping(self, key: tuple, provider) -> ParamTerm:
        """Normalize and store a provider for the hopping at ``key = (i, j, R)``."""
        # Normalize R to a tuple so the ctx label is stable/hashable in messages.
        i, j, R = key[0], key[1], tuple(key[2])
        term = ParamTerm.from_provider(provider, ctx=f"hopping[{i},{j},{R}]")
        self.hoppings[key] = term  # note: stored under the ORIGINAL key
        return term

    def discard(self, *, onsite_idx=None, hop_key=None) -> None:
        """Remove the provider at an on-site index and/or hopping key, if present.

        Used when a term is overwritten with a plain number or the orbital is
        deleted. ``dict.pop(k, None)`` is a no-op when the key isn't there, so
        this never raises on an already-absent term.
        """
        if onsite_idx is not None:
            self.onsite.pop(onsite_idx, None)
        if hop_key is not None:
            self.hoppings.pop(hop_key, None)

    @property
    def names(self) -> tuple[str, ...]:
        """Sorted union of all free parameter names across registered terms.

        e.g. if one site needs {"U"} and one hopping needs {"t1", "phi"}, this
        returns ("U", "phi", "t1") — the full set of values the user must supply.
        """
        out: set[str] = set()
        for term in self.onsite.values():
            out.update(term.names)
        for term in self.hoppings.values():
            out.update(term.names)
        return tuple(sorted(out))

    def missing(self, given: Iterable[str]) -> tuple[str, ...]:
        """Registered parameter names not covered by the given assignment names.

        Empty tuple => the caller supplied everything the model needs.
        """
        return tuple(n for n in self.names if n not in set(given))

    def copy(self) -> "ParameterRegistry":
        """Shallow copy sharing the (immutable) term objects.

        ParamTerm recipes are never mutated after creation, so the copy can
        reuse the same objects; only the dict containers are duplicated (so
        registering/discarding on the copy doesn't touch the original).
        """
        new = ParameterRegistry()
        new.onsite = dict(self.onsite)
        new.hoppings = dict(self.hoppings)
        return new


# frozen=True makes instances immutable (and hashable) — they're pure metadata.
@dataclass(frozen=True)
class AxisFD:
    """Finite-difference metadata for one swept parameter axis.

    When you sweep a parameter (e.g. ``phi=[0, 0.1, 0.2, ...]``), the velocity
    code wants dH/dphi, computed by finite differences along that axis. To do
    that it needs to know the spacing and whether the axis wraps around. This
    little record carries exactly that, for one axis.
    """

    name: str
    index: int  # which sweep axis this is (0 = first swept param, 1 = second, ...)
    step: float  # uniform spacing Δ between consecutive samples
    periodic: bool  # does the axis wrap (so the derivative can use a periodic stencil)?
    trimmed: bool  # was a duplicated wrap-around endpoint dropped (see normalize_axis)?


def normalize_axis(values, *, name: str, period: float | None = None):
    """Normalize a 1D parameter sweep and report finite-difference metadata.

    The job: validate that the samples are uniformly spaced, decide whether the
    axis is periodic, and (if a periodic axis literally repeats its first point
    at the end) drop that duplicate so we don't differentiate across a zero gap.

    Why periodicity matters: a periodic axis can use a wrap-around (central)
    stencil everywhere, which is more accurate than the one-sided differences
    you'd otherwise need at the two ends.

    Three situations are detected:

      (a) period given, endpoint duplicates start  -> trim it, periodic=True
          e.g. period=2π, values = linspace(0, 2π, N, endpoint=True)
               span = values[-1]-values[0] = 2π == period  => last point is a
               copy of the first; drop it. trimmed=True.
      (b) period given, NO duplicate but full cycle -> keep all, periodic=True
          e.g. period=2π, values = linspace(0, 2π, N, endpoint=False)
               step*N == 2π == period  => the samples already tile one full
               period with no repeat. trimmed=False.
      (c) no period given, but last == first       -> trim it, periodic=True
          We infer periodicity purely from the data repeating its endpoint.

    Returns
    -------
    values_unique : np.ndarray
        Copy of the input with any duplicated endpoint removed.
    step : float
        Uniform spacing between samples (computed BEFORE trimming).
    is_periodic : bool
        True if the sweep spans a full cycle.
    trimmed : bool
        True when the final element duplicated the first and was dropped.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        # Need at least two samples to even define a spacing / a derivative.
        raise ValueError(
            f"Parameter '{name}' must be one-dimensional with at least two samples."
        )

    # All consecutive gaps must be equal — finite differences assume a uniform grid.
    diffs = np.diff(arr)  # [v1-v0, v2-v1, ...]
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(f"Parameter '{name}' must be uniformly spaced.")
    step = float(diffs[0])  # the common spacing Δ (measured before any trimming)

    periodic = False
    trimmed = False
    if period is not None:
        # User TOLD us the period (e.g. 2π for a phase). Decide (a) vs (b).
        period = float(period)
        span = arr[-1] - arr[0]  # distance covered by the samples as given
        if np.isclose(span, period):
            # (a) last sample sits exactly one period past the first => it's a
            #     duplicate of the first under wrap-around. Drop it.
            arr = arr[:-1]
            periodic = True
            trimmed = True
        elif np.isclose(step * arr.size, period):
            # (b) N samples spaced by Δ already cover one full period (Δ*N == P)
            #     with no repeated endpoint. Periodic, but nothing to trim.
            periodic = True
    else:
        # No period supplied: only way to know it's periodic is if the data
        # itself repeats — last value equals first. (c)
        if np.isclose(arr[-1], arr[0]):
            arr = arr[:-1]
            periodic = True
            trimmed = True

    # .copy() so the caller can't mutate our (possibly sliced) view of the input.
    return arr.copy(), step, periodic, trimmed


# frozen=True: a SweepSpec is an immutable description of one call's parameters.
@dataclass(frozen=True)
class SweepSpec:
    """Scalar assignments plus cartesian sweep axes for one method call.

    Produced by :meth:`from_params` from the ``**params`` of a single
    ``hamiltonian``/``velocity`` call. It splits the parameters into:
      - ``scalars``: name -> single fixed value (applied to every grid point)
      - ``names`` + ``axes``: the swept parameters and the list of values for
        each. ``names[i]`` is swept over ``axes[i]``; all axes are combined as a
        cartesian product (every combination of values).
    """

    scalars: dict[str, object]  # fixed values, e.g. {"U": 0.3}
    names: tuple[str, ...]  # swept parameter names, e.g. ("phi", "t")
    axes: tuple[tuple, ...]  # one tuple of values per swept name (parallel to names)

    @classmethod
    def from_params(cls, params: Mapping[str, object], *, spinful: bool) -> "SweepSpec":
        """Partition call parameters by the documented shape rules (strict).

        Walks each ``name: value`` and classifies ``value`` purely by its shape
        into either a scalar assignment or a sweep axis. The shape table at the
        top of this module is the spec; anything not matching raises. ``spinful``
        gates the 2x2 / Pauli-4-vector forms, which only make sense with spin.
        """
        scalars: dict[str, object] = {}
        sweep_names: list[str] = []
        sweep_axes: list[tuple] = []

        # Single place to raise the "I don't understand this value" error, with
        # the full list of accepted forms so the user can self-correct.
        def reject(name, raw, why):
            raise ValueError(
                f"Parameter '{name}' has unsupported {why}. Accepted forms: "
                "scalar, 1-D array (sweep), (1, n) vector value, "
                "(2, 2) matrix value (spinful), (n, 4) Pauli-vector sweep (spinful)."
            )

        for name, raw in params.items():
            # Plain Python number -> a scalar. (bool is excluded: True/False are
            # ints in Python, but almost never a legitimate parameter value.)
            if isinstance(raw, (int, float, complex)) and not isinstance(raw, bool):
                scalars[name] = raw
                continue
            # Beyond plain numbers we only accept array-likes; reject e.g. str/dict.
            if not isinstance(raw, (list, tuple, np.ndarray)):
                reject(name, raw, f"type {type(raw).__name__}")

            arr = np.asarray(raw)
            if arr.ndim == 0:
                # 0-D array (e.g. np.float64(3.0)) -> unwrap to a Python scalar.
                scalars[name] = arr.item()
            elif arr.ndim == 1:
                # 1-D list/array -> a SWEEP axis (loop over these values).
                if arr.shape[0] == 0:
                    raise ValueError(
                        f"Parameter sweep '{name}' must provide at least one value."
                    )
                # Keep the ORIGINAL python elements (don't cast the array) so the
                # provider sees the user's dtype — e.g. complex stays complex.
                sweep_names.append(name)
                sweep_axes.append(tuple(arr[i] for i in range(arr.shape[0])))
            elif arr.ndim == 2 and arr.shape[0] == 1:
                # Shape (1, n): a SINGLE n-vector value (the leading 1 is just
                # "one value"), not a sweep of n scalars. Store as a scalar.
                scalars[name] = arr[0, :].copy()  # single vector value
            elif arr.ndim == 2 and arr.shape == (2, 2):
                # A single 2x2 matrix value — only meaningful for spinful models
                # (a spin operator / 2x2 block).
                if not spinful:
                    raise ValueError(
                        f"Parameter '{name}' is a 2x2 array, but the model is spinless."
                    )
                scalars[name] = arr.copy()
            elif arr.ndim == 2 and arr.shape[1] == 4:
                # Shape (n, 4): SWEEP over n Pauli 4-vectors [c0, cx, cy, cz],
                # each describing a 2x2 block c0*I + c·σ. Spinful only.
                if not spinful:
                    raise ValueError(
                        f"Parameter '{name}' has shape {arr.shape}, but the model "
                        "is spinless."
                    )
                sweep_names.append(name)
                sweep_axes.append(tuple(arr[i, :].copy() for i in range(arr.shape[0])))
            else:
                # Any other shape (e.g. (3, 3), 3-D, ...) is unsupported.
                reject(name, raw, f"shape {arr.shape}")

        return cls(scalars=scalars, names=tuple(sweep_names), axes=tuple(sweep_axes))

    @property
    def has_axes(self) -> bool:
        """True when at least one parameter is swept over an axis.

        False => this call is a single point (scalars only), so the caller can
        skip all the grid machinery and just build once.
        """
        return bool(self.axes)

    def assignment_names(self) -> tuple[str, ...]:
        """All parameter names this call assigns (scalars first, then sweeps)."""
        return tuple(self.scalars) + self.names

    def fd_axes(
        self, param_periods: Mapping[str, float] | None
    ) -> tuple[list[np.ndarray], list[AxisFD]]:
        """Float evaluation axes plus finite-difference metadata per sweep axis.

        Two parallel-ish outputs:
          - ``raw_axes``: the float values for EVERY sweep axis (used to actually
            evaluate the model on the user's grid).
          - ``specs``: an :class:`AxisFD` for each axis we can DIFFERENTIATE
            along. Axes that are not differentiable (fewer than two samples, or
            non-numeric so they can't be cast to float cleanly) appear in
            ``raw_axes`` but get NO ``AxisFD`` entry — so ``specs`` may be shorter
            than ``raw_axes``. ``AxisFD.index`` records the original axis position.
        """
        periods = dict(param_periods or {})
        raw_axes: list[np.ndarray] = []
        specs: list[AxisFD] = []
        for idx, name in enumerate(self.names):
            # Cast this axis to float for evaluation/differencing.
            axis_array = np.asarray(self.axes[idx], dtype=float)
            raw_axes.append(axis_array.copy())
            # Can't finite-difference a single point or a non-1-D axis: skip the
            # spec (but the axis still gets evaluated, hence it's in raw_axes).
            if axis_array.ndim != 1 or axis_array.size < 2:
                continue
            # Validate spacing + detect periodicity (period optional, per name).
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

        ``build_fn(assignments)`` builds the physics object(s) for ONE fully
        resolved set of parameter values and must return a TUPLE of arrays whose
        shapes don't depend on the assignment (e.g. ``(H,)`` or ``(vel, ham)``).

        This method loops over every combination of the sweep axes, calls
        ``build_fn`` for each, and stacks the results so that each swept
        parameter becomes its own array axis. ``n_lead[i]`` says how many leading
        dimensions array ``i`` already has (e.g. a k-axis), so the new sweep axes
        are inserted *after* them — keeping a consistent (k, sweeps, matrix...)
        layout. ``n_lead`` has one entry per array returned by ``build_fn``.

        Worked example (one sweep of length L=5, build_fn returns (H,) with
        H.shape == (Nk, M, M), n_lead == (1,)):
          - loop 5 times, fill stacked_flat[0] of shape (5, Nk, M, M)
          - reshape -> (5, Nk, M, M)            [axis_lengths = (5,)]
          - moveaxis axis 0 -> position 1       [lead = 1]
          - final shape (Nk, 5, M, M): the sweep sits right after the k-axis.
        With two axes (L0, L1) it'd be (Nk, L0, L1, M, M).
        """
        # ``axes`` lets a caller substitute different value lists (e.g. the
        # normalized/trimmed grid) while keeping the same names; default = own axes.
        eval_axes = self.axes if axes is None else tuple(axes)
        if len(eval_axes) != len(self.names):
            raise ValueError("Number of parameter sweep axes must match names.")
        # No sweep at all -> just build once at the scalar values and return.
        if not eval_axes:
            return tuple(build_fn(self.scalars))

        axis_lengths = [len(axis) for axis in eval_axes]  # e.g. [L0, L1]
        n_blocks = int(np.prod(axis_lengths, dtype=int))  # total grid points L0*L1*...
        stacked_flat = None  # allocated lazily once we know each block's shape/dtype
        assignments = self.scalars.copy()  # start from fixed values, add sweeps below

        # np.ndindex walks the multi-dimensional grid in row-major (C) order,
        # yielding tuples like (0,0), (0,1), ..., flattened to flat_idx 0,1,...
        for flat_idx, multi_idx in enumerate(np.ndindex(*axis_lengths)):
            # Resolve each swept name to its value at this grid point.
            for axis_idx, name in enumerate(self.names):
                assignments[name] = eval_axes[axis_idx][multi_idx[axis_idx]]

            blocks = tuple(build_fn(assignments))  # build the object(s) here
            if stacked_flat is None:
                # First iteration: allocate one flat (n_blocks, *block_shape)
                # buffer per returned array, matching its shape & dtype exactly.
                stacked_flat = [
                    np.empty((n_blocks, *block.shape), dtype=block.dtype)
                    for block in blocks
                ]
            # Drop this grid point's results into row flat_idx of each buffer.
            for out, block in zip(stacked_flat, blocks, strict=True):
                out[flat_idx] = block

        p = len(axis_lengths)  # number of sweep axes to reshape the flat dim into
        results = []
        for out, lead in zip(stacked_flat, n_lead, strict=True):
            # Un-flatten the leading n_blocks dim back into the p sweep axes:
            # (n_blocks, *rest) -> (L0, L1, ..., *rest).
            out = out.reshape(*axis_lengths, *out.shape[1:])
            # Then slide those p sweep axes (currently at front, positions 0..p)
            # to sit right after the array's own `lead` leading dims.
            results.append(np.moveaxis(out, range(p), range(lead, lead + p)))
        return tuple(results)
