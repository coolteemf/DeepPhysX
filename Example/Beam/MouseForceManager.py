import numpy as np
from time import time_ns


class MouseForceManager:

    def __init__(self, topology):
        self.neighbors = self.analyze_topology(topology)

    def analyze_topology(self, topology):
        edges = topology.edges.value
        neighbors = [[] for _ in range(len(topology.position.value))]
        for edge in edges:
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])
        return neighbors

    def find_picked_node(self, force):
        idx = np.unique(np.nonzero(force)[0])
        return None if len(idx) == 0 else idx[0]

    def find_neighbors(self, node):
        return self.neighbors[node]
