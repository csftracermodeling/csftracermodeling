import SVMTK as svmtk
import os
import meshio
import pathlib


def create_gwv_mesh(pial_stl, ventricles_stl, resolution,
                    output, remove_ventricles=True):

    assert os.path.isfile(pial_stl)
    assert os.path.isfile(ventricles_stl)

    # Create SVMTk Surfaces from STL files
    pial = svmtk.Surface(pial_stl)
    ventricles = svmtk.Surface(ventricles_stl)
    surfaces = [pial, ventricles]

    # Define identifying tags for the different regions
    tags = {"pial": 1, "ventricle": 2}

    # Define the corresponding subdomain map
    smap = svmtk.SubdomainMap()
    smap.add("10", tags["pial"])
    smap.add("01", tags["ventricle"])

    # Mesh and tag the domain from the surfaces and map
    domain = svmtk.Domain(surfaces, smap)

    domain.create_mesh(resolution)

    # Remove subdomain with right tag from the domain
    if remove_ventricles:
        domain.remove_subdomain(tags["ventricle"])

    # Save the mesh
    domain.save(output)


if __name__ == "__main__":

    outputfile = "./data/freesurfer/meshes/lh.mesh"

    os.makedirs(pathlib.Path(outputfile).parent, exist_ok=True)

    create_gwv_mesh(pial_stl="./data/freesurfer/surf/lh.pial.stl",
                    ventricles_stl="./data/freesurfer/surf/ventricles.stl",
                    resolution=16,
                    output=outputfile, remove_ventricles=True)

    os.system("meshio convert " + outputfile + " " + outputfile.replace(".mesh", ".xml"))
    os.system("meshio convert " + outputfile + " " + outputfile.replace(".mesh", ".xdmf"))

    try:
        from fenics import *
        lhmesh = Mesh(outputfile.replace(".mesh", ".xml"))

        meshboundary = BoundaryMesh(lhmesh, "exterior")

        boundarymeshfile = outputfile.replace(".mesh", "_boundary.xml")

        File(boundarymeshfile) << meshboundary

        os.system("meshio convert " + boundarymeshfile + " " + boundarymeshfile.replace(".xml", ".xdmf"))
        os.system("meshio convert " + boundarymeshfile + " " + boundarymeshfile.replace(".xml", ".stl"))

    except ModuleNotFoundError:
        pass

    print()
    print("*" * 80)
    print("Meshing done, to view the result in paraview run")
    print("paraview " + outputfile.replace(".mesh", ".xdmf"))

    print()
    print("To view the mesh surface in freeview, run")
    print("freeview ./data/freesurfer/mri/aseg.mgz -f " + boundarymeshfile.replace(".xml", ".stl"))
    print("*" * 80)
