<div align="right">

**Joint ICTP MARVEL College: Materials simulations in the age of AI**  
Thursday, 4 June 2026

&nbsp;


</div>

<div align="center">

# From Berry phases to hybrid Wannier centers with PythTB

</div>



**Tutors**: T. Cole, N. Baù, R. Dal Molin, M. Dowlatabadi, A. Marrazzo, R. D'Souza

**Time budget**: 30 minute

This is a `PythTB` tutorial about using Berry phases to diagnose topology. After a brief introduction to the PythTB workflow, we will compute **hybrid Wannier charge centers** (HWCCs) for the Haldane and Kane-Mele models and use them to read off the topological invariant in each case.

The key idea:

> A hybrid Wannier center is a Berry phase computed along one reciprocal-space direction, repeated as a function of the other reciprocal-space direction.

PythTB references:

- [Documentation](https://pythtb.readthedocs.io/en/latest/) - https://pythtb.readthedocs.io/en/latest/
- [GitHub repository](https://github.com/pythtb/pythtb) - https://github.com/pythtb/pythtb

---

## Setup

Copy and paste the following imports into your notebook to get started:

```python
import numpy as np
import matplotlib.pyplot as plt

from pythtb import Lattice, Mesh, TBModel, WFArray
```

The main PythTB objects today:

- `TBModel` — the tight-binding Hamiltonian;
- `Mesh` — the $k$-point grid;
- `WFArray` — stores Bloch eigenvectors on the mesh, and computes Berry-phase quantities from them via `WFArray.berry_phase` and `WFArray.chern_number`.

---

## 1. The PythTB workflow

> 4 minutes

A typical PythTB calculation has four objects working together. A `Lattice` describes the unit-cell geometry, a `TBModel` is the tight-binding Hamiltonian built on that lattice, a `Mesh` specifies the $k$-point sampling, and a `WFArray` stores the Bloch eigenvectors on the mesh and provides the methods we use for topological diagnostics. We will walk through each piece briefly so the pattern is familiar before the Haldane exercise.

### Building a `TBModel`

`Lattice` holds the lattice vectors in Cartesian coordinates, the orbital positions in reduced (fractional) coordinates, and a list of which lattice directions are periodic. `TBModel` then takes that `Lattice` and exposes `set_hop(amp, i, j, R)`, which adds a hopping amplitude from orbital `j` in the home cell to orbital `i` in cell `R` (with the Hermitian conjugate added automatically). A complete model definition looks like this:

```python
lat = Lattice(
    lat_vecs=[[1.0, 0.0], [0.5, np.sqrt(3)/2]],  # honeycomb lattice vectors
    orb_vecs=[[1/3, 1/3], [2/3, 2/3]],           # A and B sublattices
    periodic_dirs=[0, 1],
)

model = TBModel(lattice=lat, spinful=False)
model.set_hop(t, 0, 1, [ 0,  0])      # one call per hopping
model.set_hop(t, 0, 1, [-1,  0])
# ... more hoppings ...
model.visualize()                     # sanity check
```

> **Note**: for today you do not need to write any `set_hop` lines yourself. Both the Haldane and Kane-Mele models come pre-built as helpers in `pythtb.models`. We show the pattern here so the helper outputs make sense when you inspect them.

### `Mesh` and `WFArray`

A `Mesh` declares the parameter-space axes for our calculation. Passing `["k", "k"]` to the constructor declares a 2D Brillouin-zone mesh, and `build_grid(shape=(Nx, Ny))` fills it with a uniform $N_x \times N_y$ grid of points. We then create a `WFArray` over the same `Lattice` and `Mesh`, and call `solve_model` to diagonalize the Hamiltonian at every $k$-point on the mesh:

```python
mesh = Mesh(["k", "k"])   # one "k" per reciprocal-space direction
mesh.build_grid(shape=(51, 51))

wfa = WFArray(model.lattice, mesh)
wfa.solve_model(model)
```

After `solve_model`, the `WFArray` holds the Bloch eigenvectors at every mesh point. Its `berry_phase` method takes those eigenvectors and computes Berry phases along whichever mesh axis you choose, for whichever band(s) you choose. The function looks like this:

```python
phases = wfa.berry_phase(axis_idx: int, state_idx: list[int], berry_evals: bool)
```

The `axis_idx` specifies which mesh axis to integrate along, and `state_idx` specifies which band(s) to include in the Berry phase calculation. Setting `berry_evals=True` returns the individual eigenphases of the Wilson loop rather than their sum (this is for when there are multiple occupied bands).


Calling `berry_phase` will integrate the Berry connection along one mesh axis (`axis_idx`), and gives us the Berry phase at every value of the other axes. The result gives a curve that we will use to read off topology — the hybrid Wannier charge center (HWCC) flow. 

```python
hwcc = phases / (2 * np.pi)  # convert Berry phase to HWCC
```

This is what we will be doing for the Haldane and Kane-Mele models in the following exercises.

---

## 2. Haldane: HWCC flow and the Chern number

> 10 minutes

In a 2D insulator with broken time-reversal symmetry the topological invariant is the **Chern number**, the integral of the Berry curvature over the BZ,

$$
C = \frac{1}{2\pi} \int_{\mathrm{BZ}} F_{xy}(k_x, k_y)\, dk_x\, dk_y .
$$

A nonzero Chern number forbids fully localized Wannier functions — the **topological obstruction** to a localized Wannier representation. We can still build *hybrid Wannier functions* (localized along $x$, Bloch-extended along $y$), and the positions of their centers will be what we use to detect topology.

The diagnostic is built from a **Berry phase**: the integral of the Berry connection $A_\mu(k) = i\langle u(k) | \partial_{k_\mu} u(k) \rangle$ around a closed loop in $k$-space,

$$
\gamma = \oint A_\mu(k)\, dk_\mu.
$$

For a one-band, one-dimensional insulator, $\gamma / 2\pi$ is exactly the electronic Wannier center in reduced coordinates — this is the modern theory of polarization. In our 2D case we close the loop along $k_x$ at fixed $k_y$ and divide by $2\pi$ to get the **hybrid Wannier charge center** (HWCC) at that $k_y$,

$$
\bar{x}(k_y) = \frac{1}{2\pi} \oint A_x(k_x, k_y)\, dk_x.
$$

Tracing this for every $k_y$ gives the **HWCC flow**. Its net winding as $k_y$ traverses the BZ equals the Chern number. This is the real-space fingerprint of the topological obstruction.

**Goal**: Diagnose the topology of the Haldane model from band inversion, the Chern number, and the HWCC flow.

### Task 1: Define the Haldane model

The Haldane model is a two-band honeycomb model with a staggered sublattice potential $\Delta$, a real nearest-neighbor hopping $t_1$, and a complex next-nearest-neighbor hopping $t_2 e^{i\phi}$ that breaks time-reversal symmetry:

$$H = \Delta \sum_i (-)^i c_i^\dagger c_i + t_1 \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.}) + t_2 \sum_{\langle\langle i,j \rangle\rangle} (e^{i \phi} c_i^\dagger c_j + \text{h.c.}).
$$

The `haldane` helper builds the model for you:

```python
from pythtb.models import haldane

delta = 1.0   # staggered sublattice potential
t1    = -1.0  # nearest-neighbor hopping
t2    = -0.3  # complex next-nearest-neighbor hopping

hal = haldane(delta=delta, t1=t1, t2=t2, phi=np.pi/2)

print(hal)
hal.visualize()
```

### Task 2: Band structure and band inversion

`plot_bands` accepts a list of high-symmetry $k$-nodes in reduced coordinates and an optional `proj_orb_idx` list of orbital indices. `proj_orb_idx` then colors each band by its projection onto the set of chosen orbitals,

$$ 
P_n(k) = \sum_{i \in \mathrm{proj}} \left| \langle \psi_n(k) \mid \phi_i \rangle \right|^2.
$$

The sublattice that dominates the occupied band near the gap determines whether the bands are inverted.

```python
k_nodes = [[0, 0], [2/3, 1/3], [1/2, 1/2], [1/3, 2/3], [0, 0]]
k_node_labels = [r"$\Gamma$", r"$K$", r"$M$", r"$K'$", r"$\Gamma$"]
orbital_idx = [0]  # project onto the first orbital (A sublattice)

hal.plot_bands(
    nk=500,
    k_nodes=k_nodes,
    k_node_labels=k_node_labels,
    proj_orb_idx=orbital_idx,
)
```

**Questions**

- How can you tell from the band structure whether the system is in a topological phase?
- What is the sublattice character of the occupied band at $K$? At $\Gamma$?

### Task 3: Compute the HWCC flow

The workflow follows the same pattern as the primer. We build a 2D mesh, create and solve a `WFArray` on it, and then call `berry_phase` to integrate along $k_x$ — which returns one Berry phase per $k_y$ point on the mesh.

1. `mesh = Mesh(["k", "k"])`, then `mesh.build_grid(shape=(51, 51))`.
2. `wfa = WFArray(hal.lattice, mesh)` and `wfa.solve_model(hal)`.
3. `phases = wfa.berry_phase(axis_idx=0, state_idx=[0])`.
4. `xbar = phases / (2 * np.pi)`.

**Questions**
- What does `axis_idx` specify in the `berry_phase` call?
- What does `state_idx=[0]` specify in the `berry_phase` call?

> **Checkpoint**: at this stage `xbar` should be a 1D array with one HWCC value per $k_y$ point on the mesh.

### Task 4: Plot the HWCC flow

The HWCC is only defined modulo one lattice constant, so we plot a few periodic images stacked together to see any winding clearly. The $k_y$ values themselves are retrieved from `mesh.get_axis_range(1, 1)`, which returns the parameter values along the second mesh axis.

```python
ky = mesh.get_axis_range(1, 1)

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
for shift in range(-1, 2):
    ax.plot(ky, xbar + shift, "o-", ms=3, color="tab:blue")

ax.set(xlim=(0, 1), ylim=(-0.5, 1.5),
       xlabel=r"$k_y$", ylabel=r"$\bar{x}(k_y)$")
ax.grid(alpha=0.25)
plt.show()
```

### Task 5: Compare with the Chern number

`WFArray.chern_number` integrates the Berry curvature directly over the 2D mesh. We pass `state_idx=[0]` to select the occupied band and `plane=(0, 1)` to specify that the curvature should be integrated over the first two mesh axes.

```python
C_h = wfa.chern_number(state_idx=[0], plane=(0, 1))

print(C_h)
```

The magnitude of the Chern number should match the net winding of the HWCC flow,

$$
|\Delta\bar{x}| = |C|.
$$

### Task 6: Repeat in the trivial phase

Now reduce $|t_2|$ until the system becomes trivial. At $t_2 = 0$ you recover boron nitride (and at $\Delta = t_2 = 0$ you recover graphene). Somewhere in between, the gap closes at $K$, and that is the topological transition. Fill in the table below for your two choices of $t_2$:

| $t_2$ | HWCC endpoint change $\Delta\bar{x}$ | Occupied Chern number | Band inversion at $K$ or $K'$? | Topological or trivial? |
| --- | --- | --- | --- | --- |
| $-0.3$ <br><br> |  |  |  |  |
|  <br><br> |  |  |  |  |

---

## 3. Kane-Mele: $\mathbb{Z}_2$ from Wilson-loop eigenphases

> 10 minutes

With time-reversal symmetry the Berry curvature is odd under TR, so the Chern number vanishes — HWCC winding of a single band cannot be the invariant. The right object is the **Wilson loop**, the path-ordered exponential of the non-abelian Berry connection along $k_x$,

$$
U(k_y) = \mathcal{P} \exp\left( i \oint_0^{2\pi} A_x(k_x, k_y)\, dk_x \right),
$$

a unitary on the occupied subspace. Its eigenphases $\theta_n(k_y)$ are the centers of the two hybrid Wannier branches — the multi-band generalization of the single HWCC from Haldane, obtained from `WFArray.berry_phase(..., berry_evals=True)`.

> **Note**:
> The non-Abelian Berry connection is a matrix-valued generalization of the Berry connection that arises when there are multiple occupied bands. It is defined as $A_\mu^{mn}(k) = i\langle u_m(k) | \partial_{k_\mu} u_n(k) \rangle$, where $m$ and $n$ run over the occupied bands. The Wilson loop is then the path-ordered exponential of this matrix-valued connection, which captures the parallel transport of the entire occupied subspace along a closed loop in $k$-space.

Time-reversal symmetry forces the two branches into a Kramers pair at $k_y = 0$ and $k_y = 1/2$ (the TRIMs) but leaves them free to wander between. The $\mathbb{Z}_2$ invariant is **partner switching**: in the topological phase, the branches swap partners between the two TRIMs (any horizontal reference line over $k_y \in [0, 1/2]$ is crossed an odd number of times); in the trivial phase they don't (even number of crossings).

**Goal**: diagnose the $\mathbb{Z}_2$ topology of Kane-Mele from band inversion and the Wilson-loop eigenphase flow.

### Task 1: Define the Kane-Mele model

The Kane-Mele model is effectively two copies of Haldane, one for each spin, with opposite signs of the complex NNN hopping so that time-reversal symmetry is preserved overall. It also includes a Rashba term that breaks $S_z$ conservation:

$$H = \Delta \sum_{i} c_i^\dagger c_i + t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.}) + \lambda_{SO} \sum_{\langle\langle i,j \rangle\rangle} (c_i^\dagger \sigma_z c_j + \text{h.c.}) + \lambda_{R} \sum_{\langle i,j \rangle} (c_i^\dagger (\boldsymbol{\sigma} \times \hat{\mathbf{d}}_{ij}) c_j + \text{h.c.}).
$$

```python
from pythtb.models import kane_mele

delta  = 1.0   # staggered sublattice potential
t      = 1.0   # nearest-neighbor hopping
soc    = 0.3   # intrinsic SOC
rashba = 0.25  # Rashba SOC

km = kane_mele(delta=delta, t=t, soc=soc, rashba=rashba)

print(km)
km.visualize()
```

> **Note**: the helper sets `spinful=True` internally, and your `WFArray` must be built with the same setting.

### Task 2: Band structure

We use the same $k$-path as in Haldane, but because the model is spinful, there are now four bands instead of two. Specifying an orbital in `proj_orb_idx` implicitly sums over all spin orientations (up and down).

```python
k_nodes = [[0, 0], [2/3, 1/3], [1/2, 1/2], [1/3, 2/3], [0, 0]]
k_node_labels = [r"$\Gamma$", r"$K$", r"$M$", r"$K'$", r"$\Gamma$"]
proj_orb_idx = [0]  # project onto the first orbital (A sublattice)

km.plot_bands(
    nk=500,
    k_nodes=k_nodes,
    k_node_labels=k_node_labels,
    proj_orb_idx=proj_orb_idx,
)
```

**Question**

- Where in the Brillouin zone does the band inversion occur, and how can you tell?

### Task 3: Compute the Wilson-loop eigenphases

The pipeline is the same as the one we just used for Haldane, but the model is spinful and we ask for the individual eigenphases rather than their sum:

1. Build the `Mesh` as in the Haldane case
2. Build the `WFArray(..., spinful=True)` and populate it with the Bloch eigenstates. Since the model is spinful, we have to set `spinful=True` here.
3. Compute the Wilson-loop eigenphases with `berry_phase(..., berry_evals=True)`. What should `state_idx` be in this case? How many occupied bands are there? 

> **Tip**: `state_idx` is a list of band indices. The states are indexed in order of energy, so the lowest energy state is indexed as `0`, the next one as `1`, and so on.


4. Translate the eigenphases to HWCCs and define them as the variable `centers` for the next step.

**Question** 
- How many Wilson-loop eigenphases (HWCC branches) do you expect to see, and why?
- What shape is your `centers` array, and what does each axis represent?

### Task 4: Plot and read off $\mathbb{Z}_2$

```python
ky = mesh.get_axis_range(1, 1)

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
for shift in range(-1, 3):
    ax.plot(ky, centers[:, 0] + shift, "k.", ms=4)
    ax.plot(ky, centers[:, 1] + shift, "k.", ms=4)

ax.axvline(0.5, color="tab:red", lw=0.8)
ax.set_title(rf"Kane-Mele: $\delta={delta}$")
ax.set(xlim=(0, 1), ylim=(-1, 2),
       xlabel=r"$k_y$", ylabel=r"$\bar{x}_n(k_y)$")
ax.grid(alpha=0.25)
plt.show()
```

Now draw any horizontal reference line across the plot over $k_y \in [0, 1/2]$ and count how many times the two branches cross it. An odd number of crossings means the system is in the quantum spin Hall phase ($\mathbb{Z}_2 = 1$), while an even number (including zero) means the trivial phase.

### Task 5: Compare trivial and topological

Repeat the same calculation with a larger value of `delta` and compare the two HWCC flows. Fill in the table below:

| `delta` | Partner switching? | Band inversion? | $\quad\quad \mathbb{Z}_2 \quad\quad$ | Topological or trivial? |
| --- | :-: | :-: | :-: | :-: |
| $2.5$ <br><br>|  |  |  |  |
| <br><br>  |  |  |  |  |

---

## Wrap-up

> 4 minutes

The 1D Berry phase generalizes to the HWCCs we computed in Haldane and Kane-Mele:

| System | Berry-phase object | Topological reading |
| --- | --- | --- |
| 1D (e.g. SSH) | one Berry phase over $k$ | Wannier center / polarization |
| 2D Haldane | Berry phase over $k_x$ as a function of $k_y$ | Chern number from HWCC winding |
| 2D Kane-Mele | Wilson-loop eigenphases over $k_x$ as a function of $k_y$ | $\mathbb{Z}_2$ from partner switching |

Take-home questions:

1. What are some ways to diagnose topology?
2. Why is a HWCC a Berry phase in disguise?
3. What topological invariant does the HWCC winding compute in the Haldane model?
4. Why does Kane-Mele require Wilson-loop eigenphases rather than a single Berry phase?
5. How is time-reversal symmetry reflected in the HWCC flow of Kane-Mele?

<div align="right">
&nbsp;

&nbsp;
---
**Hands-on by Trey Cole**

Thursday, 4 June 2026 - Trieste, Italy

</div>
