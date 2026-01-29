from pythtb.mesh import Mesh
from pythtb.wannier import Wannier
from pythtb.wfarray import WFArray
from pythtb.models import fu_kane_mele


def test_spinful_project():
    model = fu_kane_mele(1, 1)
    tf_list = [[(0, 0, 1), (1, 0, -1)], [(0, 1, 1), (1, 1, -1)]]

    nks = [4, 4, 4]

    mesh = Mesh(dim_k=3, axis_types=["k", "k", "k"])
    mesh.build_grid(shape=nks)
    wfa = WFArray(model.lattice, mesh, spinful=model.spinful)
    wfa.solve_model(model)

    wannier = Wannier(wfa)
    wannier.project(tf_list)
    assert wannier.tilde_states.nstates == 2
    assert wannier.tilde_states.spinful is True
    assert wannier.tilde_states.wfs.shape == (4, 4, 4, 2, 2, 2)
    tilde_states = wannier.tilde_states.states(flatten_spin_axis=True)

    assert tilde_states.shape == (4, 4, 4, 2, 4)
