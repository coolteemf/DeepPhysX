import numpy as np
from vedo import Mesh


class GridMapping:

    def __init__(self, source: Mesh, target: Mesh):
        """
        Python implementation of a barycentric mapping between a source mesh and a target mesh.
        """

        # Clone meshes
        self.source = source.clone()
        self.target = target.clone()

        # Compute barycentric coordinates
        self.bar_coord, self.cells = self.init_mapping()

    def init_mapping(self):
        """
        Compute barycentric coordinates of target to map source.
        """

        # 2. CLOSEST POINT FROM SOURCE FOR EACH POINT OF TARGET

        # 2.3 Find the closest points
        cells_centers = self.source.points()[self.source.cells()].mean(axis=1)

        D = np.array([np.linalg.norm(cells_centers - p, axis=1) for p in self.target.points()])

        cells = np.array(self.source.cells())[np.argmin(D, axis=1)]

        # 3. BARYCENTRIC COORDINATES

        # 3.1 Define hexahedrons
        bar = []
        for p, cell in zip(self.target.points(), cells):
            hexa = self.source.points()[np.array(cell)]
            anchor = np.min(hexa, axis=0)
            corner = np.max(hexa, axis=0)
            size = corner - anchor
            b = (p - anchor) / size
            bar.append(b)

        return bar, cells

    def apply(self, source_positions):
        """
        Apply mapping between new source positions and target with barycentric coordinates.
        """

        H = source_positions[self.cells][:, [0, 4, 7, 3, 1, 5, 6, 2]]
        b = np.array(self.bar_coord)
        one = np.ones_like(b[:, 0])
        P = ((H[:, 6].T * b[:, 0] + H[:, 2].T * (one - b[:, 0])) * b[:, 1] +
             (H[:, 5].T * b[:, 0] + H[:, 1].T * (one - b[:, 0])) * (one - b[:, 1])) * b[:, 2] + \
            ((H[:, 7].T * b[:, 0] + H[:, 3].T * (one - b[:, 0])) * b[:, 1] +
             (H[:, 4].T * b[:, 0] + H[:, 0].T * (one - b[:, 0])) * (one - b[:, 1])) * (one - b[:, 2])

        return Mesh([P.T, self.target.cells()])
