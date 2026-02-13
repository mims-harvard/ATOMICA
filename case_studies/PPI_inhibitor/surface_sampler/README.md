# Surface Sampler for Protein Chains

This package provides tools for generating molecular surface meshes from PDB/CIF files and sampling points uniformly from those surfaces.

## Features

- **Chain Extraction**: Extract specific chains from PDB or CIF files
- **Surface Mesh Generation**: Generate molecular surface using MSMS
- **Point Sampling**: Uniformly sample points from mesh surfaces
- **Format Support**: Works with both PDB and CIF file formats
- **Multi-character Chain IDs**: Handles CIF chain IDs with multiple characters (e.g., "AAA")

## Installation Requirements

- Python 3.9+
- BioPython
- NumPy
- MSMS binary (located at `/n/home13/afang/.conda/envs/masif_env3/bin/msms`)

## Main Functions

### Function 1: `get_mesh()`

Generate a surface mesh for a specific chain from a PDB/CIF file.

```python
from surface_sampler import get_mesh

mesh_data = get_mesh(
    pdb_cif_path='5DB2.pdb',
    chain_id='A',
    output_path='output_mesh.ply',
    density=3.0,           # vertices per Angstrom^2
    probe_radius=1.5       # probe sphere radius in Angstroms
)

print(f"Generated {len(mesh_data['vertices'])} vertices")
print(f"Generated {len(mesh_data['faces'])} faces")
```

**Parameters:**
- `pdb_cif_path` (str): Path to PDB or CIF file
- `chain_id` (str): Chain ID to extract and process
- `output_path` (str, optional): Path to save mesh as PLY file
- `density` (float): MSMS vertex density (default: 3.0)
- `probe_radius` (float): Probe sphere radius (default: 1.5)

**Returns:** Dictionary containing:
- `vertices`: Nx3 numpy array of vertex coordinates
- `faces`: Mx3 numpy array of triangle face indices
- `normals`: Nx3 numpy array of vertex normals
- `atom_ids`: List of atom identifiers
- `areas`: Dictionary of surface areas per atom
- `mesh_path`: Path to saved PLY file (if output_path provided)

### Function 2: `sample_points()`

Sample points uniformly from a mesh surface.

```python
from surface_sampler import sample_points

points_data = sample_points(
    mesh_path='output_mesh.ply',
    num_points=1000,
    output_path='sampled_points.xyz',
    seed=42  # for reproducibility
)

print(f"Sampled {len(points_data['points'])} points")
```

**Parameters:**
- `mesh_path` (str): Path to PLY mesh file
- `num_points` (int): Number of points to sample
- `output_path` (str, optional): Path to save points as XYZ file
- `seed` (int, optional): Random seed for reproducibility

**Returns:** Dictionary containing:
- `points`: (num_points, 3) numpy array of point coordinates
- `points_path`: Path to saved XYZ file (if output_path provided)

### Convenience Function: `get_mesh_and_sample()`

Combines both functions in a single call.

```python
from surface_sampler import get_mesh_and_sample

results = get_mesh_and_sample(
    pdb_cif_path='5DB2.pdb',
    chain_id='A',
    num_points=1000,
    mesh_output_path='mesh.ply',
    points_output_path='points.xyz'
)
```

## Usage Examples

### Example 1: PDB File with Single-Character Chain ID

```python
from surface_sampler import get_mesh, sample_points

# Generate mesh
mesh_data = get_mesh('5DB2.pdb', 'A', 'mesh.ply')

# Sample 1000 points
points_data = sample_points('mesh.ply', 1000, 'points.xyz')
```

### Example 2: CIF File with Multi-Character Chain ID

```python
from surface_sampler import get_mesh_and_sample

# Generate mesh and sample points in one step
results = get_mesh_and_sample(
    '7BBP.cif',
    'AAA',  # Multi-character chain ID
    num_points=1000,
    mesh_output_path='mesh.ply',
    points_output_path='points.xyz'
)
```

### Example 3: In-Memory Processing

```python
from surface_sampler import get_mesh
from sample_surface_points import sample_surface_points

# Get mesh without saving
mesh_data = get_mesh('5DB2.pdb', 'A')

# Sample points directly from arrays
points = sample_surface_points(
    mesh_data['vertices'],
    mesh_data['faces'],
    num_points=1000
)
```

## Module Descriptions

### `extract_chain.py`
Extracts specific chains from PDB/CIF files using BioPython. Handles both file formats and automatically converts multi-character chain IDs to single characters for PDB format compatibility.

### `pdb_to_xyzrn.py`
Converts PDB files to XYZRN format required by MSMS. Includes Van der Waals radii for standard atom types and filters atoms appropriately.

### `compute_surface_mesh.py`
Wraps the MSMS binary to compute molecular surfaces. Handles file I/O and parses MSMS output files (.vert and .face).

### `sample_surface_points.py`
Implements area-weighted sampling to uniformly distribute points across the mesh surface. Uses barycentric coordinates for sampling within triangles.

### `surface_sampler.py`
Main API providing high-level functions for mesh generation and point sampling.

## Testing

Run the test suite to verify the installation:

```bash
python test_surface_sampler.py
```

The test suite includes:
1. Test with PDB file: Chain A from 5DB2.pdb
2. Test with CIF file: Chain AAA from 7BBP.cif

Both tests should pass with output showing mesh statistics and sampled points.

## Test Results

```
Test 1 (PDB - 5DB2 Chain A): PASSED
  Vertices: 52,130
  Faces: 104,264
  Total surface area: 18,031.76 Angstrom^2

Test 2 (CIF - 7BBP Chain AAA): PASSED
  Vertices: 18,245
  Faces: 36,494
  Total surface area: 6,469.83 Angstrom^2
```

## File Formats

### PLY (Polygon File Format)
Mesh files are saved in ASCII PLY format containing:
- Vertex coordinates (x, y, z)
- Vertex normals (nx, ny, nz)
- Face indices (triangles)

### XYZ Format
Point files are saved in XYZ format:
```
<number of points>
<comment line>
C <x1> <y1> <z1>
C <x2> <y2> <z2>
...
```

## Implementation Notes

- **MSMS Parameters**: Default density is 3.0 vertices/Angstrom^2 and probe radius is 1.5 Angstroms, following MaSIF conventions
- **Sampling Method**: Uses area-weighted sampling to ensure uniform point distribution
- **Chain ID Mapping**: Multi-character chain IDs from CIF files are automatically mapped to single characters for PDB format compatibility
- **Temporary Files**: Automatically cleaned up after processing

## Credits

Adapted from the MaSIF implementation by Pablo Gainza (LPDI STI EPFL 2019) with modifications for standalone use and CIF file support.

## License

Released under an Apache License 2.0
