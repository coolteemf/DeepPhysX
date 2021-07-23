import math
import random
import numpy as np
import Sofa.SofaBaseTopology


def compute_grid_resolution(max_bbox, min_bbox, cell_size):
    sx = math.fabs(max_bbox[0] - min_bbox[0])  # Size in the x-axis
    sy = math.fabs(max_bbox[1] - min_bbox[1])  # Size in the y-axis
    sz = math.fabs(max_bbox[2] - min_bbox[2])  # Size in the z-axis
    s = np.array([sx, sy, sz])
    center = np.array(min_bbox) + s / 2.
    s = s * 1.25

    # New bounding box
    new_min_bbox = (center - s / 2.).tolist()
    new_max_bbox = (center + s / 2.).tolist()
    print("BBOX [{}, {}] (sx={}, sy={}, sz={})".format(new_min_bbox, new_max_bbox, sx, sy, sz))

    # Grid
    cell_size = cell_size * min(sx, sy, sz)  # Cell size of the grid is x% of the object's size
    print("Cell size = {}x{}x{}".format(cell_size, cell_size, cell_size))

    nx = int(sx / cell_size)  # Number of cells in the x-axis
    ny = int(sy / cell_size)  # Number of cells in the y-axis
    nz = int(sz / cell_size)  # Number of cells in the z-axis
    print("Nx = {}, Ny = {}, Nz = {}".format(nx, ny, nz))

    number_of_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    print("Number of nodes in regular grid = {}".format(number_of_nodes))

    return [nx + 1, ny + 1, nz + 1]


def extract_visible_nodes(camera_position, normals, positions, dot_thresh=0.0, rand_thresh=0.95,
                          distance_from_camera_thresh=500):
    indices_visible_surface = [i for i in range(normals.shape[0]) if
                               np.dot(normals[i], camera_position) < dot_thresh and random.random() > rand_thresh]
    indices_front_surface = []
    for node in indices_visible_surface:
        if np.linalg.norm(positions[node] - camera_position) < distance_from_camera_thresh:
            indices_front_surface.append(node)
    indices_front_surface = np.unique(np.ravel(indices_front_surface))
    return positions[indices_front_surface].tolist()


def from_sparse_to_regular_grid(nb_nodes_regular_grid, sparse_grid, sparse_grid_MO):
    # Initialize mapping between sparse grid and regular grid
    initial_positions_sparse_grid = sparse_grid_MO.position.array()
    indices_sparse_to_regular = np.zeros(initial_positions_sparse_grid.shape[0], dtype=np.int32)
    indices_regular_to_sparse = np.full(nb_nodes_regular_grid, -1, dtype=np.int32)
    for i in range(initial_positions_sparse_grid.shape[0]):
        idx = sparse_grid.getRegularGridNodeIndex(i)
        indices_sparse_to_regular[i] = idx
        indices_regular_to_sparse[idx] = i
    # Recover rest shape positions of sparse grid in regular grid
    regular_grid_rest_shape_position = np.zeros((nb_nodes_regular_grid, 3), dtype=np.double)
    regular_grid_rest_shape_position[indices_sparse_to_regular] = sparse_grid_MO.rest_position.array()
    return indices_sparse_to_regular, indices_regular_to_sparse, regular_grid_rest_shape_position
