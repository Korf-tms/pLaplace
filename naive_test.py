import pyvista
import numpy as np
from dolfinx import plot
import ufl
import dolfinx as dfx
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

show_pictures = True

# load mesh and domains data
# https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
with dfx.io.XDMFFile(MPI.COMM_WORLD, 'circle_with_hole.xdmf', 'r') as mesh_file:
    mesh = mesh_file.read_mesh(name='Grid')
    cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)  # taken from the link, why is it here?

# prepare tags for boudary conditions
with dfx.io.XDMFFile(MPI.COMM_WORLD, 'circle_with_hole_boundary.xdmf', 'r') as boundary_file:
    facet_tags = boundary_file.read_meshtags(mesh, name='Grid')


k = 1

# solve a little test problem to test the coefficients and boundaries
V = dfx.fem.functionspace(mesh, ('Lagrange', 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(k * ufl.grad(u), ufl.grad(v))*ufl.dx
L = dfx.fem.Constant(mesh, dfx.default_scalar_type(2.0e1))*v*ufl.dx

# set bcs
outer_facets = facet_tags.find(1)  # magical number given by the mesh construction, =max(facet_tags.values)
inner_facets = facet_tags.find(2)  # magical number given by the mesh construction, =min(facet_tags.values)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)  # taken from the link, why is it here?
inner_facet_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, inner_facets)
outer_facets_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, outer_facets)
bcs = [dfx.fem.dirichletbc(dfx.default_scalar_type(0e1), outer_facets_dofs, V),
        dfx.fem.dirichletbc(dfx.default_scalar_type(0e2), inner_facet_dofs, V)]

problem = LinearProblem(a, L, bcs=bcs, petsc_options={'ksp_type: preonly,'
                                                        'pc_type': 'lu'})
uh = problem.solve()

if show_pictures:
    grid_uh = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
    grid_uh.point_data["u"] = uh.x.array.real
    grid_uh.set_active_scalars("u")
    p2 = pyvista.Plotter()
    p2.title = 'Solution'
    p2.add_mesh(grid_uh, show_edges=True, scalar_bar_args={'vertical': True})
    p2.show()
    warped = grid_uh.warp_by_scalar()
    p3 = pyvista.Plotter()
    p3.title = 'Solution in 3d'
    p3.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalar_bar_args={'vertical': True})
    p3.show()