"""
sample_surface_points.py: Sample points uniformly from a triangular mesh surface
Uses area-weighted sampling to ensure uniform distribution across the surface.
"""
import numpy as np


def compute_triangle_areas(vertices, faces):
    """
    Compute the area of each triangle in the mesh.

    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of face indices

    Returns:
        np.ndarray: M-length array of triangle areas
    """
    # Get vertices for each triangle
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Area = 0.5 * |edge1 x edge2|
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


def sample_point_in_triangle(v0, v1, v2):
    """
    Uniformly sample a random point within a triangle.

    Args:
        v0, v1, v2 (np.ndarray): 3D coordinates of triangle vertices

    Returns:
        np.ndarray: 3D coordinates of sampled point
    """
    # Use barycentric coordinates for uniform sampling
    # See: https://math.stackexchange.com/questions/18686
    r1 = np.random.random()
    r2 = np.random.random()

    # Ensure point is inside triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Compute point using barycentric coordinates
    point = u * v0 + v * v1 + w * v2

    return point


def sample_surface_points(vertices, faces, num_points, seed=None):
    """
    Sample points uniformly from the surface of a triangular mesh.

    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of face indices
        num_points (int): Number of points to sample
        seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: (num_points, 3) array of sampled point coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute triangle areas
    areas = compute_triangle_areas(vertices, faces)

    # Normalize areas to get probability distribution
    area_probs = areas / areas.sum()

    # Sample triangles proportional to their area
    sampled_face_indices = np.random.choice(
        len(faces),
        size=num_points,
        p=area_probs,
        replace=True
    )

    # Sample one point from each selected triangle
    sampled_points = np.zeros((num_points, 3))

    for i, face_idx in enumerate(sampled_face_indices):
        # Get triangle vertices
        v0 = vertices[faces[face_idx, 0]]
        v1 = vertices[faces[face_idx, 1]]
        v2 = vertices[faces[face_idx, 2]]

        # Sample point in triangle
        sampled_points[i] = sample_point_in_triangle(v0, v1, v2)

    return sampled_points


def load_ply_mesh(ply_path):
    """
    Load mesh from PLY file.

    Args:
        ply_path (str): Path to PLY file

    Returns:
        tuple: (vertices, faces)
            - vertices: Nx3 numpy array of vertex coordinates
            - faces: Mx3 numpy array of face indices
    """
    vertices = []
    faces = []
    num_vertices = 0
    num_faces = 0
    reading_vertices = False
    reading_faces = False

    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse header
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line == 'end_header':
                reading_vertices = True
                continue

            # Read vertices
            if reading_vertices and len(vertices) < num_vertices:
                parts = line.split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(vertices) == num_vertices:
                    reading_vertices = False
                    reading_faces = True

            # Read faces
            elif reading_faces and len(faces) < num_faces:
                parts = line.split()
                # Skip first element (number of vertices in face)
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return np.array(vertices), np.array(faces, dtype=int)


def save_points_xyz(points, output_path):
    """
    Save sampled points to XYZ file format.

    Args:
        points (np.ndarray): Nx3 array of point coordinates
        output_path (str): Path to output XYZ file
    """
    with open(output_path, 'w') as f:
        f.write(f"{len(points)}\n")
        f.write("Sampled surface points\n")
        for point in points:
            f.write(f"C {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

    return output_path
