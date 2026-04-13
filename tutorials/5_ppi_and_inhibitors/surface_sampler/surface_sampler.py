"""
surface_sampler.py: Main API for protein surface mesh generation and point sampling
Provides two main functions as specified in TODO.md:
1. get_mesh: Generate mesh from PDB/CIF file and chain ID
2. sample_points: Sample points from mesh surface
"""
import os
import tempfile
try:
    from .extract_chain import extract_chain
    from .compute_surface_mesh import compute_msms, save_mesh_ply
    from .sample_surface_points import sample_surface_points, load_ply_mesh, save_points_xyz
except ImportError:
    from extract_chain import extract_chain
    from compute_surface_mesh import compute_msms, save_mesh_ply
    from sample_surface_points import sample_surface_points, load_ply_mesh, save_points_xyz


def get_mesh(pdb_cif_path, chain_id, output_path=None, density=3.0, probe_radius=1.5):
    """
    Generate a surface mesh for a specific chain from a PDB/CIF file.

    Args:
        pdb_cif_path (str): Path to PDB or CIF file
        chain_id (str): Chain ID to extract and process
        output_path (str, optional): Path to save mesh as PLY file. If None, mesh is not saved.
        density (float): MSMS vertex density (vertices per Angstrom^2). Default: 3.0
        probe_radius (float): Probe sphere radius in Angstroms. Default: 1.5

    Returns:
        dict: Dictionary containing:
            - 'vertices': Nx3 numpy array of vertex coordinates
            - 'faces': Mx3 numpy array of triangle face indices
            - 'normals': Nx3 numpy array of vertex normals
            - 'atom_ids': List of atom identifiers
            - 'areas': Dictionary of surface areas per atom
            - 'mesh_path': Path to saved PLY file (if output_path was provided)

    Example:
        >>> mesh_data = get_mesh('5DB2.pdb', 'A', 'output_mesh.ply')
        >>> print(f"Generated mesh with {len(mesh_data['vertices'])} vertices")
    """
    # Create temporary file for extracted chain
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
        tmp_pdb_path = tmp_file.name

    try:
        # Step 1: Extract the chain
        extract_chain(pdb_cif_path, tmp_pdb_path, chain_id)

        # Step 2: Compute surface mesh using MSMS
        vertices, faces, normals, atom_ids, areas = compute_msms(
            tmp_pdb_path,
            density=density,
            probe_radius=probe_radius
        )

        # Step 3: Save mesh if output path is provided
        mesh_path = None
        if output_path is not None:
            mesh_path = save_mesh_ply(vertices, faces, normals, output_path)

        result = {
            'vertices': vertices,
            'faces': faces,
            'normals': normals,
            'atom_ids': atom_ids,
            'areas': areas,
            'mesh_path': mesh_path
        }

        return result

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)


def sample_points(mesh_path, num_points, output_path=None, seed=None):
    """
    Sample points uniformly from a mesh surface.

    Args:
        mesh_path (str): Path to PLY mesh file
        num_points (int): Number of points to sample from the surface
        output_path (str, optional): Path to save sampled points as XYZ file. If None, not saved.
        seed (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary containing:
            - 'points': (num_points, 3) numpy array of sampled point coordinates
            - 'points_path': Path to saved XYZ file (if output_path was provided)

    Example:
        >>> points_data = sample_points('output_mesh.ply', 1000, 'sampled_points.xyz')
        >>> print(f"Sampled {len(points_data['points'])} points from surface")
    """
    # Load mesh from PLY file
    vertices, faces = load_ply_mesh(mesh_path)

    # Sample points from the surface
    points = sample_surface_points(vertices, faces, num_points, seed=seed)

    # Save points if output path is provided
    points_path = None
    if output_path is not None:
        points_path = save_points_xyz(points, output_path)

    result = {
        'points': points,
        'points_path': points_path
    }

    return result


def get_mesh_and_sample(pdb_cif_path, chain_id, num_points,
                        mesh_output_path=None, points_output_path=None,
                        density=3.0, probe_radius=1.5, seed=None):
    """
    Convenience function that combines get_mesh and sample_points.

    Args:
        pdb_cif_path (str): Path to PDB or CIF file
        chain_id (str): Chain ID to extract and process
        num_points (int): Number of points to sample from the surface
        mesh_output_path (str, optional): Path to save mesh PLY file
        points_output_path (str, optional): Path to save sampled points XYZ file
        density (float): MSMS vertex density. Default: 3.0
        probe_radius (float): Probe sphere radius. Default: 1.5
        seed (int, optional): Random seed for reproducibility

    Returns:
        dict: Combined results from mesh generation and point sampling

    Example:
        >>> results = get_mesh_and_sample('5DB2.pdb', 'A', 1000,
        ...                               'mesh.ply', 'points.xyz')
        >>> print(f"Mesh: {len(results['vertices'])} vertices")
        >>> print(f"Points: {len(results['points'])} sampled points")
    """
    # Generate mesh
    mesh_data = get_mesh(pdb_cif_path, chain_id, mesh_output_path, density, probe_radius)

    # Sample points
    if mesh_output_path is not None:
        # Use saved mesh file
        points_data = sample_points(mesh_output_path, num_points, points_output_path, seed)
    else:
        # Use in-memory mesh data
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as tmp_file:
            tmp_mesh_path = tmp_file.name

        try:
            save_mesh_ply(mesh_data['vertices'], mesh_data['faces'],
                         mesh_data['normals'], tmp_mesh_path)
            points_data = sample_points(tmp_mesh_path, num_points, points_output_path, seed)
        finally:
            if os.path.exists(tmp_mesh_path):
                os.remove(tmp_mesh_path)

    # Combine results
    result = {**mesh_data, **points_data}
    return result
