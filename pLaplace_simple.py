import ufl
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx as dfx
from dolfinx.fem import Constant
from ufl import grad, inner, dx
from dolfinx import nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element

import pyvista



if __name__ == '__main__':
    square_size = 1
    mesh_size = 20
    lagrange_element_order = 1

    mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD,
                                     [[0.0, 0.0], [square_size, square_size]],
                                     [mesh_size, mesh_size],
                                     cell_type=dfx.mesh.CellType.triangle)
    

    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'circle_with_hole.xdmf', 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')
        cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)  # taken from the link, why is it here?

    # prepare tags for boudary conditions
    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'circle_with_hole_boundary.xdmf', 'r') as boundary_file:
        facet_tags = boundary_file.read_meshtags(mesh, name='Grid')


    x = ufl.SpatialCoordinate(mesh)

    P_n = element("Lagrange", mesh.basix_cell(), lagrange_element_order)
    Q = dfx.fem.functionspace(mesh, P_n)
    v = ufl.TestFunction(Q)
    u = dfx.fem.Function(Q)

    boundary_facets = dfx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = dfx.fem.dirichletbc(Constant(mesh, 0.0),
                             dfx.fem.locate_dofs_topological(Q, 1, boundary_facets), Q)
    
    p = 5
    E = 1.0/p*inner(grad(u), grad(u))**(p/2.0)*dx 
    E += - x[0]*u*dx
    # E += - x[0]*(1-x[0])*(1-x[1])*x[1]*u*dx
    # F = inner(grad(u), grad(v))*dx
    F = ufl.derivative(E, u)
    J = ufl.derivative(F, u)

    u_init_expression = dfx.fem.Expression(x[0]*(1-x[0])*(1-x[1])*x[1], Q.element.interpolation_points())
    u.interpolate(u_init_expression)

    problem = NonlinearProblem(F, u, bcs=[bc], J=J)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True
    solver.max_it = 200

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.INFO)
    n, converged = solver.solve(u)
    assert (converged)
    print(f"Number of interations: {n:d}")

    grid_uh = pyvista.UnstructuredGrid(*dfx.plot.vtk_mesh(Q ))
    grid_uh.point_data["u"] = u.x.array.real
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