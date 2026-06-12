"""Tests for pythtb.parameters: terms, registry, and sweep specs."""

import numpy as np
import pytest

from pythtb import TBModel, Lattice
from pythtb.parameters import ParamTerm, SweepSpec, normalize_axis


def two_site_model(spinful=False):
    lat = Lattice(
        [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]], periodic_dirs=[0, 1]
    )
    return TBModel(lat, spinful=spinful)


# ---------------------------------------------------------------- ParamTerm
def test_expression_term_free_names_and_eval():
    term = ParamTerm.from_provider("t1 * np.exp(1j * phi)", ctx="hopping[0,1,(0, 0)]")
    assert term.names == ("phi", "t1")
    val = term.evaluate({"t1": 2.0, "phi": np.pi})
    assert np.isclose(val, -2.0)


def test_negated_name_is_an_expression():
    term = ParamTerm.from_provider("-d", ctx="onsite[1]")
    assert term.names == ("d",)
    assert term.evaluate({"d": 0.7}) == -0.7


def test_invalid_expression_fails_at_registration():
    with pytest.raises(ValueError, match="not a valid Python expression"):
        ParamTerm.from_provider("t1*", ctx="hopping[0,1,(0, 0)]")


def test_expression_missing_value_is_helpful():
    term = ParamTerm.from_provider("t1*0.5", ctx="hopping[0,1,(0, 0)]")
    with pytest.raises(ValueError, match="needs a value for parameter 't1'"):
        term.evaluate({})


def test_callable_signature_resolved_once():
    term = ParamTerm.from_provider(lambda t, phi=0.0: t, ctx="hopping[0,1,(0, 0)]")
    assert term.names == ("t",)  # only required params are tracked
    assert term.evaluate({"t": 1.5, "unrelated": 9}) == 1.5


# ------------------------------------------------------- expressions on models
def test_model_with_expression_providers():
    tb = two_site_model()
    tb.set_onsite(["d", "-d"])
    tb.set_hop("t1", 0, 1, [0, 0])
    tb.set_hop("t1*0.5", 0, 0, [1, 0])
    names = {n for entry in tb.parameters for n in entry["names"]}
    assert names == {"d", "t1"}

    k = tb.k_uniform_mesh([3, 3])
    evals = tb.solve_ham(k, d=0.4, t1=1.0)
    assert evals.shape == (9, 2)

    # expression terms appear in the Hamiltonian with the right value:
    # onsite d - (-d) = 0.8, plus the t1*0.5 self-hop at k=0 (+ c.c.) = 1.0
    H = tb.hamiltonian(np.array([[0.0, 0.0]]), d=0.4, t1=1.0)
    assert np.isclose(H[0, 0, 0].real - H[0, 1, 1].real, 0.8 + 1.0)


# ------------------------------------------------------------------ SweepSpec
def test_partition_scalars_and_axes():
    spec = SweepSpec.from_params({"a": 1.0, "b": [0.0, 0.5, 1.0]}, spinful=False)
    assert spec.scalars == {"a": 1.0}
    assert spec.names == ("b",)
    assert len(spec.axes[0]) == 3


def test_partition_rejects_bad_shapes():
    with pytest.raises(ValueError, match="unsupported shape"):
        SweepSpec.from_params({"a": np.zeros((3, 5))}, spinful=False)
    with pytest.raises(ValueError, match="unsupported shape"):
        SweepSpec.from_params({"a": np.zeros((2, 2, 2))}, spinful=False)


def test_partition_pauli_requires_spinful():
    with pytest.raises(ValueError, match="spinless"):
        SweepSpec.from_params({"a": np.zeros((3, 4))}, spinful=False)
    spec = SweepSpec.from_params({"a": np.zeros((3, 4))}, spinful=True)
    assert spec.names == ("a",)


def test_fd_axes_metadata_periodic_trim():
    vals = np.linspace(0.0, 2 * np.pi, 7)  # endpoint duplicates start mod 2pi
    spec = SweepSpec.from_params({"phi": vals}, spinful=False)
    _, specs = spec.fd_axes({"phi": 2 * np.pi})
    (fd,) = specs
    assert fd.periodic and fd.trimmed
    assert np.isclose(fd.step, vals[1] - vals[0])


def test_normalize_axis_rejects_nonuniform():
    with pytest.raises(ValueError, match="uniformly spaced"):
        normalize_axis([0.0, 0.1, 0.3], name="x")


def test_evaluate_grid_axis_placement():
    spec = SweepSpec.from_params(
        {"a": [1.0, 2.0], "b": [10.0, 20.0, 30.0]}, spinful=False
    )

    def build(assign):
        return (np.full((4, 2, 2), assign["a"] * assign["b"]),)

    (out,) = spec.evaluate(build, n_lead=(1,))
    assert out.shape == (4, 2, 3, 2, 2)  # sweep axes after the first axis
    assert np.allclose(out[:, 1, 2], 2.0 * 30.0)
