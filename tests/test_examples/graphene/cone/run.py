import numpy as np
from pythtb import tb_model, wf_array

def graphene_model() -> tb_model:
    "Return a graphene-like model on a triangular lattice."

    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    my_model = tb_model(2, 2, lat, orb)

    return my_model


def run():
    my_model: tb_model = graphene_model()

    delta = -0.1 
    t = -1.0

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    circ_step = 31
    circ_center = np.array([1.0/3.0,2.0/3.0])
    circ_radius = 0.05

    w_circ = wf_array(my_model,[circ_step])

    for i in range(circ_step):
        ang = 2.0*np.pi*float(i)/float(circ_step-1)
        kpt=np.array([np.cos(ang)*circ_radius,np.sin(ang)*circ_radius])
        kpt+=circ_center
        w_circ.solve_on_one_point(kpt,i)
    w_circ[-1]=w_circ[0]

    berry_phase_circ_0 = w_circ.berry_phase([0],0)
    berry_phase_circ_1 = w_circ.berry_phase([1],0)
    berry_phase_circ_01 = w_circ.berry_phase([0,1],0)

    square_step=31
    square_center=np.array([1.0/3.0,2.0/3.0])
    square_length=0.1

    w_square=wf_array(my_model,[square_step,square_step])

    all_kpt=np.zeros((square_step,square_step,2))
    for i in range(square_step):
        for j in range(square_step):
            kpt=np.array([square_length*(-0.5+float(i)/float(square_step-1)),
                        square_length*(-0.5+float(j)/float(square_step-1))])        
            kpt+=square_center
            all_kpt[i,j,:]=kpt
            (eval,evec)=my_model.solve_one(kpt,eig_vectors=True)
            w_square[i,j]=evec

    berr_flux_square_0 = w_square.berry_flux([0])
    berr_flux_square_1 = w_square.berry_flux([1])
    berr_flux_square_01 = w_square.berry_flux([0,1])
    plaq = w_square.berry_flux([0], individual_phases=True)

    return (berry_phase_circ_0, berry_phase_circ_1, berry_phase_circ_01,
            berr_flux_square_0, berr_flux_square_1, berr_flux_square_01, plaq)
