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


gmsh.initialize()
gmsh.model.add("Domain_with_hole")

r1 = 10  # Radius of the larger circle
r2 = 1   # Radius of the hole

mesh_size_outer = 1  # Mesh size for the outer circle
mesh_size_inner = 0.5  # Mesh size for the inner circle

assert(r1 > r2)

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

# Create a distance field for the outer circle
outer_dist_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(outer_dist_field, "CurvesList", [outer_circle])
gmsh.model.mesh.field.setNumber(outer_dist_field, "NumPointsPerCurve", 100)

# Create a distance field for the inner circle
inner_dist_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inner_dist_field, "CurvesList", [inner_circle])
gmsh.model.mesh.field.setNumber(inner_dist_field, "NumPointsPerCurve", 100)

# Create a threshold field for the outer circle
outer_threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(outer_threshold_field, "InField", outer_dist_field)
gmsh.model.mesh.field.setNumber(outer_threshold_field, "SizeMin", mesh_size_outer)
gmsh.model.mesh.field.setNumber(outer_threshold_field, "SizeMax", mesh_size_outer)
gmsh.model.mesh.field.setNumber(outer_threshold_field, "DistMin", 0)
gmsh.model.mesh.field.setNumber(outer_threshold_field, "DistMax", r1)

# Create a threshold field for the inner circle
inner_threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inner_threshold_field, "InField", inner_dist_field)
gmsh.model.mesh.field.setNumber(inner_threshold_field, "SizeMin", mesh_size_inner)
gmsh.model.mesh.field.setNumber(inner_threshold_field, "SizeMax", mesh_size_inner)
gmsh.model.mesh.field.setNumber(inner_threshold_field, "DistMin", 0)
gmsh.model.mesh.field.setNumber(inner_threshold_field, "DistMax", r2)

# Combine the threshold fields
min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [outer_threshold_field, inner_threshold_field])

# Set the combined field as the background mesh size field
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.mesh.generate(dim=2)

# Save the mesh to a file
gmsh.write("circle_with_hole.msh")

# Finalize gmsh
gmsh.finalize()


create_xdmf_mesh_from_msh_with_cell_data("circle_with_hole")
