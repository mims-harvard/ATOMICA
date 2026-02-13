"""
example_usage.py: Example usage of the surface sampler API

This script demonstrates the basic usage of the two main functions:
1. get_mesh() - Generate surface mesh from PDB/CIF file
2. sample_points() - Sample points from the mesh surface
"""
from surface_sampler import get_mesh, sample_points, get_mesh_and_sample


def example_1_separate_steps():
    """Example 1: Using get_mesh and sample_points separately"""
    print("Example 1: Separate mesh generation and point sampling")
    print("-" * 60)

    # Step 1: Generate mesh
    pdb_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/datasets/2p2idb/2p2idb_2024-04-08_pdbs/5DB2.pdb"
    mesh_data = get_mesh(
        pdb_cif_path=pdb_path,
        chain_id='A',
        output_path='example_mesh.ply'
    )
    print(f"Mesh generated: {len(mesh_data['vertices'])} vertices, {len(mesh_data['faces'])} faces")

    # Step 2: Sample points
    points_data = sample_points(
        mesh_path='example_mesh.ply',
        num_points=500,
        output_path='example_points.xyz',
        seed=42  # for reproducibility
    )
    print(f"Sampled {len(points_data['points'])} points")
    print(f"Point coordinates range: {points_data['points'].min():.2f} to {points_data['points'].max():.2f}")
    print()


def example_2_combined():
    """Example 2: Using get_mesh_and_sample for convenience"""
    print("Example 2: Combined mesh generation and point sampling")
    print("-" * 60)

    # Generate mesh and sample points in one call
    cif_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/datasets/2p2idb/2p2idb_2024-04-08_cifs/7BBP.cif"
    results = get_mesh_and_sample(
        pdb_cif_path=cif_path,
        chain_id='AAA',
        num_points=500,
        mesh_output_path='example_mesh_combined.ply',
        points_output_path='example_points_combined.xyz',
        seed=42
    )

    print(f"Mesh: {len(results['vertices'])} vertices, {len(results['faces'])} faces")
    print(f"Points: {len(results['points'])} sampled points")
    print(f"Total surface area: {sum(results['areas'].values()):.2f} Angstrom^2")
    print()


def example_3_in_memory():
    """Example 3: Working with in-memory data without saving to disk"""
    print("Example 3: In-memory processing (no file output)")
    print("-" * 60)

    from sample_surface_points import sample_surface_points

    # Generate mesh without saving
    pdb_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/datasets/2p2idb/2p2idb_2024-04-08_pdbs/5DB2.pdb"
    mesh_data = get_mesh(pdb_path, 'A')

    # Sample points directly from arrays
    points = sample_surface_points(
        vertices=mesh_data['vertices'],
        faces=mesh_data['faces'],
        num_points=500,
        seed=42
    )

    print(f"Generated mesh with {len(mesh_data['vertices'])} vertices")
    print(f"Sampled {len(points)} points without saving to disk")
    print(f"Points shape: {points.shape}")
    print()


def example_4_custom_parameters():
    """Example 4: Using custom MSMS parameters"""
    print("Example 4: Custom MSMS parameters")
    print("-" * 60)

    pdb_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/datasets/2p2idb/2p2idb_2024-04-08_pdbs/5DB2.pdb"

    # High-resolution mesh (higher density)
    mesh_high_res = get_mesh(
        pdb_path, 'A',
        density=5.0,  # Higher density = more vertices
        probe_radius=1.4
    )

    # Low-resolution mesh (lower density)
    mesh_low_res = get_mesh(
        pdb_path, 'A',
        density=1.5,  # Lower density = fewer vertices
        probe_radius=1.5
    )

    print(f"High-resolution mesh: {len(mesh_high_res['vertices'])} vertices")
    print(f"Low-resolution mesh: {len(mesh_low_res['vertices'])} vertices")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Surface Sampler API Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_1_separate_steps()
    example_2_combined()
    example_3_in_memory()
    example_4_custom_parameters()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
