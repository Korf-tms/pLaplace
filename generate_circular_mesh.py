import gmsh
import meshio


def create_xdmf_mesh_from_msh_with_cell_data(filename):
    # based on
    # https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html#read-in-msh-files-with-dolfinx
    # and
    # https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
    
    mesh_in = meshio.read(f"{filename}.msh")
    
    cell_types = ["triangle", "line"]
    filename_suffixes = {"triangle": '', "line": "boundary"}
    points = mesh_in.points[:, :2]  # cut the z-coordinates for 2d mesh TODO: use as parameter
    for cell_type in cell_types:
        cell_data = mesh_in.get_cell_data("gmsh:physical", cell_type)
        cells = mesh_in.get_cells_type(cell_type)
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"Domains": [cell_data]})
        name = f"{filename}_{filename_suffixes[cell_type]}" if filename_suffixes[cell_type] else f"{filename}"
        meshio.write(f"{name}.xdmf", out_mesh)
        print(f"Mesh written to files: {name}.xdmf, {name}.h5")


def create_circular_mesh(r1, r2, name="circle_with_hole"):
    assert (r1 > r2)

    gmsh.initialize()
    gmsh.model.add("Domain_with_hole")

    outer_circle = gmsh.model.occ.addCircle(0, 0, 0, r1)
    inner_circle = gmsh.model.occ.addCircle(0, 0, 0, r2)

    outer_loop = gmsh.model.occ.addCurveLoop([outer_circle])
    inner_loop = gmsh.model.occ.addCurveLoop([inner_circle])

    gmsh.model.occ.synchronize()

    surface = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.occ.synchronize()

    outer_tag = gmsh.model.addPhysicalGroup(1, [outer_circle])
    inner_tag = gmsh.model.addPhysicalGroup(1, [inner_circle])
    gmsh.model.set_physical_name(1, outer_tag, name="OuterBoundary")
    gmsh.model.set_physical_name(1, inner_tag, name="InnerBoundary")

    domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.set_physical_name(2, domain_tag, name="DiskWithHole")

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)
    gmsh.model.mesh.generate()
    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.remove_duplicate_elements()

    gmsh.model.mesh.generate(dim=2)

    # Save the mesh to a file
    gmsh.write(f"{name}.msh")

    # Finalize gmsh
    gmsh.finalize()

    create_xdmf_mesh_from_msh_with_cell_data(f"{name}")


if __name__ == "__main__":
    outer_r = 1
    inner_r = 0.4
    create_circular_mesh(outer_r, inner_r)
