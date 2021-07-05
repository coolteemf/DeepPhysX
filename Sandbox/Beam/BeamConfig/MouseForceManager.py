"""
MouseForceManager.py
Component used to avoid too local forces on simulated objects by:
        1. Scaling the applied local force if the amplitude is too big
        2. Applying forces on neighbors nodes with a gaussian distribution
"""

import numpy as np
import math


class MouseForceManager:

    def __init__(self, topology, max_force, surface):
        self.neighbors, self.positions = self.analyze_topology(topology)
        self.max_force = max_force
        self.surface = surface.quads.value.reshape(-1)

    def analyze_topology(self, topology):
        """
        Get the positions of each node of the topology; find the neighbors of each node of the topology.
        :param topology: Sofa Topology
        :return: list of neighbors, list of positions
        """
        positions, edges = topology.position.value, topology.edges.value
        neighbors = [[] for _ in range(len(positions))]
        for edge in edges:
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])
        return neighbors, positions

    def find_picked_node(self, forces):
        """
        Returns the index of the picked node (non zero force).
        :param forces: forces applied on the simulated object
        :return: index of node
        """
        idx = np.unique(np.nonzero(forces)[0])
        return None if len(idx) == 0 else idx[0]

    def scale_max_force(self, forces):
        """
        Scales the max force if it exceeds the threshold.
        :param forces: forces applied on the simulated object
        :return:
        """
        forces = np.copy(forces)
        scales = [abs(forces[i]) / self.max_force[i] for i in range(3)]
        scale = max(scales)
        if scale > 1:
            forces = [forces[i] / scale for i in range(3)]
        return forces

    def distribute_force(self, node, forces, gamma, radius):
        """
        Distribute the force in the defined area with a gaussian distribution.
        :param node: index of picked node
        :param forces: forces applied on the simulated object
        :param gamma: compression of the Gaussian distribution
        :param radius: radius of the area around the picked node where forces will be applied
        :return: forces, index of nodes in area
        """
        # Get the nodes in the area defined by radius and on the surface of the object
        nodes_in_area = self.find_neighbors(node, radius)
        nodes_in_area = list(set(nodes_in_area).intersection(set(self.surface)))
        max_force = forces[node]
        if len(nodes_in_area) != 0:
            center = np.array(self.positions[node])
            # Compute a force for each node, decreasing with the distance
            for n in nodes_in_area:
                position = self.positions[n]
                dist = np.linalg.norm(position - center) * 10e-2
                forces[n] = max_force * math.exp(-gamma * math.pow(dist, 2))
        if node not in nodes_in_area:
            forces[node] = [0, 0, 0]
        return forces, nodes_in_area

    def find_neighbors(self, node, radius):
        """
        Returns the nodes in the area defined around the picked node.
        :param node: index of picked node
        :param radius: radius of the defined area
        :return: list of index of nodes in the area
        """
        # The picked node is in the center of the area
        nodes_in_area = [node]
        center = np.array(self.positions[node])
        # Init the list of potential-nodes-in-area with the neighbors of the picked node
        potential_nodes = self.neighbors[node]
        for potential_node in potential_nodes:
            position = np.array(self.positions[potential_node])
            # Check if the potential node is in the area
            if np.linalg.norm(position - center) < radius:
                nodes_in_area.append(potential_node)
                # Add the neighbors of this node to the list of potential-nodes-in-area
                for potential_node_neighbor in self.neighbors[potential_node]:
                    if potential_node_neighbor not in potential_nodes:
                        potential_nodes.append(potential_node_neighbor)
        return nodes_in_area
