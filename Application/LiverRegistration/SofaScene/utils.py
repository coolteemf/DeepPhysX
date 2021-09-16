import math
import random
import numpy as np
import Sofa.SofaBaseTopology    # Absolutely necessary to map sparse and regular grids


def compute_grid_resolution(max_bbox, min_bbox, cell_size, print_log=False):
    """
    Compute the grid resolution from the desired cell size and the grid dimensions.

    :param list max_bbox: Max upper corner of the grid
    :param list min_bbox: Min lower corner of the grid
    :param float cell_size: Desired cell size
    :param bool print_log: Print info
    :return: Number of nodes for each direction of the Grid
    """

    # Absolute size values along 3 dimensions
    sx = math.fabs(max_bbox[0] - min_bbox[0])
    sy = math.fabs(max_bbox[1] - min_bbox[1])
    sz = math.fabs(max_bbox[2] - min_bbox[2])

    # Compute number of nodes in the grid
    cell_size = cell_size * min(sx, sy, sz)  # Cells need to be hexahedron
    nx = int(sx / cell_size)
    ny = int(sy / cell_size)
    nz = int(sz / cell_size)

    if print_log:
        print(f"Cell size = {cell_size}x{cell_size}x{cell_size}")
        print(f"Nx = {nx}, Ny = {ny}, Nz = {nz}")
        number_of_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        print(f"Number of nodes in regular grid = {number_of_nodes}")

    return [nx + 1, ny + 1, nz + 1]


def from_sparse_to_regular_grid(nb_nodes_regular_grid, sparse_grid, sparse_grid_mo):
    """
    Map the indices of nodes in the sparse grid with the indices of nodes in the regular grid.

    :param int nb_nodes_regular_grid: Total number of nodes in the regular grid
    :param sparse_grid: SparseGridTopology containing the sparse grid topology
    :param sparse_grid_mo: MechanicalObject containing the positions of the nodes in the sparse grid
    :return: Mapped indices from sparse to regular grids, Mapped indices from regular to sparse regular grid,
    Rest shape positions of the regular grid
    """

    # Initialize mapping between sparse grid and regular grid
    positions_sparse_grid = sparse_grid_mo.position.array()
    indices_sparse_to_regular = np.zeros(positions_sparse_grid.shape[0], dtype=np.int32)
    indices_regular_to_sparse = np.full(nb_nodes_regular_grid, -1, dtype=np.int32)

    # Map the indices of each node iteratively
    for i in range(positions_sparse_grid.shape[0]):
        # In Sofa, a SparseGrid in computed from a RegularGrid, just use the dedicated method to retrieve their link
        idx = sparse_grid.getRegularGridNodeIndex(i)
        indices_sparse_to_regular[i] = idx      # Node i in SparseGrid corresponds to node idx in RegularGrid
        indices_regular_to_sparse[idx] = i      # Node idx in RegularGrid corresponds to node i in SparseGrid

    # Recover rest shape positions of sparse grid nodes in the regular grid
    regular_grid_rest_shape_positions = np.zeros((nb_nodes_regular_grid, 3), dtype=np.double)
    regular_grid_rest_shape_positions[indices_sparse_to_regular] = sparse_grid_mo.rest_position.array()

    return indices_sparse_to_regular, indices_regular_to_sparse, regular_grid_rest_shape_positions


def extract_visible_nodes(camera_position, object_position, normals, positions, dot_thresh=0.0, rand_thresh=0.0,
                          distance_from_camera_thresh=500):
    """
    Return the visible nodes of an object given the camera position and orientation.

    :param np.array camera_position: Coordinates of the camera
    :param np.array object_position: Coordinates of the targeted point on the object
    :param np.array normals: Normals to the surface of the object
    :param np.array positions: Positions of the points on the object's surface
    :param float dot_thresh: Threshold too distant normals from camera orientation (default: 0.0)
    :param float rand_thresh: Threshold to randomly select visible points (default: 0.0)
    :param float distance_from_camera_thresh: Threshold too distant points coordinates from camera position
    (default: inf)
    :return: Indices of visible nodes, Positions of visible nodes
    """

    # Scalar product to filter visible nodes
    idx_filtered_nodes = [i for i in range(normals.shape[0]) if
                         np.dot(normals[i], (camera_position - object_position)) > dot_thresh
                         and random.random() > rand_thresh]

    # Check the distance to the camera
    idx_visible_nodes = []
    for node in idx_filtered_nodes:
        if np.linalg.norm(positions[node] - camera_position) < distance_from_camera_thresh:
            idx_visible_nodes.append(node)
    idx_visible_nodes = np.unique(np.ravel(idx_visible_nodes))

    return idx_visible_nodes.tolist(), positions[idx_visible_nodes].tolist()


def translate_visible_nodes(visible_nodes, translation):
    for node in visible_nodes:
        for i in range(3):
            node[i] += translation[i]
    return visible_nodes
