import vedo
import math


def find_fixed_box(source_file, scale):
    """
    Find the fixed box of the model.

    :param str source_file: Mesh file
    :param float scale: Scale to apply
    :return:
    """

    # Get the bounding box
    b_min, b_max, _ = define_bbox(source_file, 0, scale)

    # Fix along the largest dimension
    sizes = b_max - b_min
    dim = sizes.tolist().index(sizes.max(0))

    # Fix the bottom of the Armadillo
    b_min[dim] -= 0.05 * sizes[dim]
    b_max[dim] = b_min[dim] + 0.1 * sizes[dim]

    return b_min.tolist() + b_max.tolist()


def find_extremities(source_file, scale):
    """
    Find the different extremities of the model.

    :param str source_file: Mesh file
    :param float scale: Scale to apply
    :return:
    """

    # Get the coordinates of the mesh
    mesh = vedo.Mesh(source_file).scale(scale)
    coords = mesh.points().copy()

    # Get the size of the bounding box
    b_min, b_max, _ = define_bbox(source_file, 0, scale)
    sizes = b_max - b_min

    # Find the tail
    tail = coords[coords[:, 2].argmax()].tolist()

    # Find the hands
    right = coords[coords[:, 0] >= sizes[0] / 3]
    left = coords[coords[:, 0] <= -sizes[0] / 3]
    r_hand = right[right[:, 2].argmin()].tolist()
    l_hand = left[left[:, 2].argmin()].tolist()

    # Find the ears
    right = coords[coords[:, 0] >= 0]
    left = coords[coords[:, 0] <= 0]
    r_ear = right[right[:, 1].argmax()].tolist()
    l_ear = left[left[:, 1].argmax()].tolist()

    # Find the muzzle
    middle = coords[coords[:, 0] >= -sizes[0] / 3]
    middle = middle[middle[:, 0] <= sizes[0] / 3]
    muzzle = middle[middle[:, 2].argmin()].tolist()

    return [tail, r_hand, l_hand, r_ear, l_ear, muzzle]


def define_bbox(source_file, margin_scale, scale):
    """
    Find the bounding box of the model.

    :param str source_file: Mesh file
    :param float scale: Scale to apply
    :param float margin_scale: Margin in percents of the bounding box
    :return: List of coordinates defined by xmin, ymin, zmin, xmax, ymax, zmax
    """

    # Find min and max corners of the bounding box
    mesh = vedo.Mesh(source_file).scale(scale)
    bbox_min = mesh.points().min(0)
    bbox_max = mesh.points().max(0)

    # Apply a margin scale to the bounding box
    bbox_min -= margin_scale * (bbox_max - bbox_min)
    bbox_max += margin_scale * (bbox_max - bbox_min)

    return bbox_min, bbox_max, bbox_min.tolist() + bbox_max.tolist()


def compute_grid_resolution(max_bbox, min_bbox, cell_size, print_log=False):
    """
    Compute the grid resolution from the desired cell size and the grid dimensions.

    :param list max_bbox: Max upper corner of the grid
    :param list min_bbox: Min lower corner of the grid
    :param float cell_size: Desired cell size
    :param bool print_log: Print info
    :return: List of grid resolution for each dimension
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

    # Print grid infos
    if print_log:
        print(f"Cell size = {cell_size}x{cell_size}x{cell_size}")
        print(f"Nx = {nx}, Ny = {ny}, Nz = {nz}")
        number_of_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        print(f"Number of nodes in regular grid = {number_of_nodes}")

    return [nx + 1, ny + 1, nz + 1]


def get_nb_nodes(source_file):
    """
    Get the number of nodes of a mesh.

    :param str source_file: Mesh file
    :return:
    """

    return vedo.Mesh(source_file).N()
