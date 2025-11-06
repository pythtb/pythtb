from pythtb.models import graphene
import matplotlib.pyplot as plt

# Create graphene TB model and visualize
model = graphene(delta=0, t=-1)
fig, ax = model.visualize()
plt.show()

# Plot band structure along high-symmetry points
nodes = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1 / 2], [0, 0]]
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
fig, ax = model.plot_bands(k_nodes=nodes, k_node_labels=label, nk=200)
plt.show()
