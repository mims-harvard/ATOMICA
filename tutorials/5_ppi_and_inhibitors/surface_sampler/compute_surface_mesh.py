"""
compute_surface_mesh.py: Compute molecular surface mesh using MSMS
Adapted from MaSIF implementation by Pablo Gainza - LPDI STI EPFL 2019
"""
import os
import numpy as np
from subprocess import Popen, PIPE
import tempfile
import random

try:
    from .pdb_to_xyzrn import pdb_to_xyzrn
except ImportError:
    from pdb_to_xyzrn import pdb_to_xyzrn


# MSMS binary location. Override with the MSMS_BIN environment variable
# (e.g. ``export MSMS_BIN=/path/to/msms``) or edit this default. MSMS can
# be obtained from https://ccsb.scripps.edu/msms/downloads/.
MSMS_BIN = os.environ.get("MSMS_BIN", "msms")


def read_msms(file_root):
    """
    Read MSMS output files (.vert and .face).

    Args:
        file_root (str): Base path for MSMS output files (without extension)

    Returns:
        tuple: (vertices, faces, normals, atom_ids)
            - vertices: Nx3 array of vertex coordinates
            - faces: Mx3 array of triangle face indices
            - normals: Nx3 array of vertex normals
            - atom_ids: List of atom identifiers for each vertex
    """
    # Read vertex file
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices from header
    header = meshdata[2].split()
    count_vertices = int(header[0])

    # Initialize arrays
    vertices = np.zeros((count_vertices, 3))
    normals = np.zeros((count_vertices, 3))
    atom_ids = [""] * count_vertices

    # Read vertex data
    for i in range(3, 3 + count_vertices):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normals[vi][0] = float(fields[3])
        normals[vi][1] = float(fields[4])
        normals[vi][2] = float(fields[5])
        atom_ids[vi] = fields[7]

    # Read face file
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of faces from header
    header = meshdata[2].split()
    count_faces = int(header[0])

    # Initialize face array
    faces = np.zeros((count_faces, 3), dtype=int)

    # Read face data (MSMS indices are 1-based, convert to 0-based)
    for i in range(3, 3 + count_faces):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1

    return vertices, faces, normals, atom_ids


def compute_msms(pdb_file, density=3.0, probe_radius=1.5, tmp_dir=None):
    """
    Compute molecular surface mesh using MSMS.

    Args:
        pdb_file (str): Path to PDB file
        density (float): Vertex density (vertices per Angstrom^2). Default: 3.0
        probe_radius (float): Probe sphere radius in Angstroms. Default: 1.5
        tmp_dir (str): Directory for temporary files. Default: system temp dir

    Returns:
        tuple: (vertices, faces, normals, atom_ids, areas)
            - vertices: Nx3 numpy array of vertex coordinates
            - faces: Mx3 numpy array of triangle face indices (0-indexed)
            - normals: Nx3 numpy array of vertex normals
            - atom_ids: List of atom identifiers
            - areas: Dictionary mapping atom IDs to their surface areas
    """
    # Create temporary directory for MSMS files
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()

    randnum = random.randint(1, 10000000)
    file_base = os.path.join(tmp_dir, f"msms_{randnum}")
    out_xyzrn = file_base + ".xyzrn"

    # Convert PDB to XYZRN format
    pdb_to_xyzrn(pdb_file, out_xyzrn)

    # Run MSMS
    FNULL = open(os.devnull, 'w')
    args = [
        MSMS_BIN,
        "-density", str(density),
        "-hdensity", str(density),
        "-probe", str(probe_radius),
        "-if", out_xyzrn,
        "-of", file_base,
        "-af", file_base
    ]

    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    # Check if MSMS ran successfully
    if p2.returncode != 0:
        raise RuntimeError(f"MSMS failed with return code {p2.returncode}\nStderr: {stderr.decode()}")

    # Read MSMS output
    vertices, faces, normals, atom_ids = read_msms(file_base)

    # Read surface areas
    areas = {}
    area_file = file_base + ".area"
    if os.path.exists(area_file):
        with open(area_file) as ses_file:
            next(ses_file)  # Skip header line
            for line in ses_file:
                fields = line.split()
                if len(fields) >= 4:
                    areas[fields[3]] = float(fields[1])

    # Clean up temporary files
    for ext in ['.area', '.xyzrn', '.vert', '.face']:
        filepath = file_base + ext
        if os.path.exists(filepath):
            os.remove(filepath)

    return vertices, faces, normals, atom_ids, areas


def save_mesh_ply(vertices, faces, normals, output_path):
    """
    Save mesh to PLY format.

    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of face indices
        normals (np.ndarray): Nx3 array of vertex normals
        output_path (str): Path to output PLY file
    """
    with open(output_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices with normals
        for i in range(len(vertices)):
            f.write(f"{vertices[i][0]} {vertices[i][1]} {vertices[i][2]} ")
            f.write(f"{normals[i][0]} {normals[i][1]} {normals[i][2]}\n")

        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    return output_path
