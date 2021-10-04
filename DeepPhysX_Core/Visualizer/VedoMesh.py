import vedo
import numpy as np
from sys import maxsize as MAX_INT


from DeepPhysX_Core.Visualizer.VedoObject import VedoObject


class VedoMesh(VedoObject):

    def __init__(self, visualizer):
        VedoObject.__init__(self, visualizer)

    def createObjects(self, scene):
        """

        :param scene: A scene is a set of objects.
        :return:
        """

        meshes = []
        scene_shapes = {}

        # Check if the positions fields are in the scene
        for key in ['positions', 'positions_shape']:
            if key not in scene:
                raise ValueError(f"[{self.name}] Field {key} is missing to create the object.")
        # Reshape the position field
        positions_shape = np.array(scene['positions_shape'], dtype=int)
        positions = scene['positions'].reshape(positions_shape)
        scene_shapes['positions'] = positions_shape

        # Check if the cell fields are in the scene
        cells = np.array([[] for _ in range(len(positions))])
        if 'cells' not in scene:
            print(f"[{self.name}] Warning: Field 'cells' is missing to create a mesh, object will be a point cloud.")
        elif 'cells_shape' not in scene:
            raise ValueError(f"[{self.name}] Field 'cells_shape' is missing to create the object.")
        else:
            cells_shape = np.array(scene['cells_shape'], dtype=int)
            cells = scene['cells'].reshape(cells_shape)
            scene_shapes['cells'] = cells_shape
            if len(cells) != len(positions):
                raise ValueError(f"[{self.name}] Number of cell arrays mismatch number of position arrays.")

        # Set the window in which the scene will be rendered
        at = np.array(scene['at'], dtype=int) if 'at' in scene else [MAX_INT for _ in range(len(positions))]
        if len(at) != len(positions):
            raise ValueError(f"[{self.name}] Number of 'at' values mismatch number of position arrays.")

        # Set the scalar field
        # Todo: not tested yet
        field_dict = scene['field_dict'] if 'field_dict' in scene else [{'scalar_field': None} for _ in range(len(positions))]

        # Create meshes
        for i in range(len(positions)):
            mesh, mesh_data = self.__createObject(position=positions[i], cell=cells[i], at=at[i], field_dict=field_dict[i])
            meshes.append(mesh)
            self.visualizer.objects[mesh] = mesh_data

        return meshes, scene_shapes

    def __createObject(self, position, cell, at, field_dict):
        """

        :param position:
        :param cell:
        :param at:
        :param field_dict:
        :return:
        """
        mesh = vedo.Mesh([position, cell])
        # Colormap
        if 'scalar_field' in field_dict and field_dict['scalar_field'] is not None:
            if 'color_map' not in field_dict or field_dict['colormap'] is None:
                field_dict['color_map'] = self.visualizer.colormap
            mesh.cmap(field_dict['color_map'])
        # Attach each field to the Mesh object
        mesh_data = {'positions': position, 'positions_shape': np.array(position.shape, dtype=int),
                     'cells': cell, 'cells_shape': np.array(cell.shape, dtype=int), 'at': self.addView(at)}
        for data_field in field_dict:
            mesh_data[data_field] = field_dict[data_field]
        return mesh, mesh_data

    def updateObjects(self):
        """

        :return:
        """
        for obj in self.visualizer.objects:
            if self.visualizer.objects[obj]['positions'] is not None:
                shape = np.array(self.visualizer.objects[obj]['positions_shape'], dtype=int)
                obj.points(self.visualizer.objects[obj]['positions'].reshape(shape))
            if 'scalar_field' in self.visualizer.objects[obj] and self.visualizer.objects[obj]['scalar_field'] is not None:
                obj.cmap(self.visualizer.objects[obj]['color_map'], self.visualizer.objects[obj]['scalar_field'])

    def createObjectData(self, positions, cells, at=None, data_dict=None):
        """
        Used from the environment to template the data dict to send.

        :param positions: List of arrays which contain the coordinates of the meshes
        :param cells: List of arrays which contain the topology of the meshes
        :param at: Index of window in which render the object
        :param data_dict: Return of a previous call to this method
        :return:
        """
        # Check the data_dict format
        if data_dict is not None:
            for key in ['positions', 'positions_shape', 'cells', 'cells_shape']:
                if key not in data_dict:
                    raise ValueError(f"[{self.name}] Field {key} is missing in given data_dict.")
        else:
            data_dict = {}
        # Convert in float arrays
        positions = np.array([positions], dtype=float)
        cells = np.array([cells], dtype=float)
        at = np.array([at], dtype=float) if at is not None else None
        # Fill data_dict (shapes are stored since arrays will be flatten when sent to TcpIpServer)
        data_dict['positions'] = positions if 'positions' not in data_dict \
            else np.concatenate((data_dict['positions'], positions))
        data_dict['positions_shape'] = np.array(data_dict['positions'].shape, dtype=float)
        data_dict['cells'] = cells if 'cells' not in data_dict \
            else np.concatenate((data_dict['cells'], cells))
        data_dict['cells_shape'] = np.array(data_dict['cells'].shape, dtype=float)
        if at is not None:
            data_dict['at'] = at if 'at' not in data_dict else np.concatenate((data_dict['at'], at))

        return data_dict

    def updateObjectData(self, positions, data_dict=None):
        """
        Used from the environment to template the updated data dict to send.

        :param data_dict: Empty dictionary or return of a previous call to this method
        :param positions: Array which contains the updated coordinates of the mesh
        :return:
        """

        # Check the data_dict format
        if data_dict is not None:
            if 'positions' not in data_dict:
                raise ValueError(f"[{self.name}] Field 'position' is missing in given data_dict.")
        else:
            data_dict = {}
        # Convert positions to float and concatenate if the field exists
        positions = np.array([positions], dtype=float)
        data_dict['positions'] = positions if 'positions' not in data_dict \
            else np.concatenate((data_dict['positions'], positions))

        return data_dict
