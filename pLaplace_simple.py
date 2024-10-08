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


def pLaplace_zero_dirichlet(mesh, p, element_type="Lagrange", order=1, rhs=None):
    x = ufl.SpatialCoordinate(mesh)

    P_n = element(element_type, mesh.basix_cell(), order)
    Q = dfx.fem.functionspace(mesh, P_n)
    v = ufl.TestFunction(Q)
    u = dfx.fem.Function(Q)

    # take from tutorial page; marks all boundary entities
    boundary_facets = dfx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.full(x.shape[1], True, dtype=bool))

    bc = dfx.fem.dirichletbc(Constant(mesh, 0.0),
                             dfx.fem.locate_dofs_topological(Q, 1, boundary_facets), Q)

    # energy functional for pLaplace
    E = 1.0 / p * inner(grad(u), grad(u)) ** (p / 2.0) * dx
    # TODO: custom rhs as input parameter?
    if rhs is not None:
        E += - rhs * u * dx
    else:
        E += - 100*ufl.sin(x[0]*x[1]) * u * dx

    F = ufl.derivative(E, u)
    J = ufl.derivative(F, u)

    # TODO: what is a good initial guess?
    # TODO: how to integrate as input to this function?
    u_init_expression = dfx.fem.Expression(x[0] * (2 - x[0]) * (2 - x[1]) * x[1], Q.element.interpolation_points())
    u.interpolate(u_init_expression)

    problem = NonlinearProblem(F, u, bcs=[bc], J=J)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-7
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
    assert converged
    return u


def plot_solution(u_function, plot_in_3d=False):
    Q = u_function.function_space
    grid_uh = pyvista.UnstructuredGrid(*dfx.plot.vtk_mesh(Q))
    grid_uh.point_data["u"] = u_function.x.array.real
    grid_uh.set_active_scalars("u")
    p2 = pyvista.Plotter()
    p2.show_bounds()
    p2.title = 'Solution'
    p2.add_mesh(grid_uh, show_edges=True, scalar_bar_args={'vertical': True})
    p2.show()

    if plot_in_3d:
        warped = grid_uh.warp_by_scalar()
        p3 = pyvista.Plotter()
        p3.show_bounds()
        p3.title = 'Solution in 3d'
        p3.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalar_bar_args={'vertical': True})
        p3.show()


def inverse_iteration_naive(mesh, p, element_type="Lagrange", order=1):
    u = None
    for counter in range(101):
        u = pLaplace_zero_dirichlet(mesh=mesh, p=p, element_type=element_type, order=order, rhs=u)
        # u.x.array[:] = u.x.array[:] / np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form(inner(u, u) * dx)))
        if counter % 10 == 0:
            print(f'Inverse iteration {counter}')
            energy = dfx.fem.assemble_scalar(dfx.fem.form(1.0 / p * inner(grad(u), grad(u)) ** (p / 2.0) * dx))
            l2_norm = np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form(inner(u, u) * dx, )))
            print(f'Energy: {energy/l2_norm}')
            plot_solution(u, True)


def test(p=4):
    # prepare meshes
    # square mesh
    square_size = 2
    mesh_size = 40
    mesh1 = dfx.mesh.create_rectangle(MPI.COMM_WORLD,
                                     [[0.0, 0.0], [square_size, square_size]],
                                     [mesh_size, mesh_size],
                                     cell_type=dfx.mesh.CellType.triangle)

    # circular mesh generated by gpt_gmsh_mesh.py
    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'circle_with_hole.xdmf', 'r') as mesh_file:
        mesh2 = mesh_file.read_mesh(name='Grid')

    for mesh in [mesh1, mesh2]:
        u = pLaplace_zero_dirichlet(mesh, p)
        plot_solution(u, True)


if __name__ == '__main__':
    # square mesh
    square_size = 2
    mesh_size = 40
    mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD,
                                     [[0.0, 0.0], [square_size, square_size]],
                                     [mesh_size, mesh_size],
                                     cell_type=dfx.mesh.CellType.triangle)


    p = 4
    inverse_iteration_naive(mesh, p)
