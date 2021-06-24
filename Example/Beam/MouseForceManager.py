import numpy as np
import math


class MouseForceManager:

    def __init__(self, topology, max_force, surface):
        if max_force is None:
            max_force = [1., 1., 1.]
        self.neighbors, self.positions = self.analyze_topology(topology)
        self.max_force = max_force
        self.surface = surface.quads.value.reshape(-1)

    def analyze_topology(self, topology):
        positions = topology.position.value
        edges = topology.edges.value
        neighbors = [[] for _ in range(len(positions))]
        for edge in edges:
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])
        return neighbors, positions

    def find_picked_node(self, force):
        idx = np.unique(np.nonzero(force)[0])
        return None if len(idx) == 0 else idx[0]

    def find_neighbors(self, node, radius):
        nodes_in_area = [node]
        center = np.array(self.positions[node])
        potential_nodes = self.neighbors[node]
        for p_n in potential_nodes:
            pos = np.array(self.positions[p_n])
            if np.linalg.norm(pos - center) < radius:
                nodes_in_area.append(p_n)
                for p_n_n in self.neighbors[p_n]:
                    if p_n_n not in potential_nodes:
                        potential_nodes.append(p_n_n)
        return nodes_in_area

    def scale_max_force(self, force):
        force = np.copy(force)
        scales = [abs(force[i]) / self.max_force[i] for i in range(3)]
        scale = max(scales)
        if scale > 1:
            force = [force[i] / scale for i in range(3)]
        return force

    def distribute_force(self, node, forces, gamma, radius):
        nodes_in_area = self.find_neighbors(node, radius)
        nodes_in_area = list(set(nodes_in_area).intersection(set(self.surface)))
        max_force = forces[node]
        if len(nodes_in_area) != 0:
            center = np.array(self.positions[node])
            for n in nodes_in_area:
                pos = self.positions[n]
                dist = np.linalg.norm(pos - center) * 10e-2
                forces[n] = max_force * math.exp(-gamma * math.pow(dist, 2))
        if node not in nodes_in_area:
            forces[node] = [0, 0, 0]
        return forces, nodes_in_area
