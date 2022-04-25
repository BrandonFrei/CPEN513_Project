import unittest
from copy import deepcopy
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import K_Partitioning as part

class TestStringMethods(unittest.TestCase):
    def test_output_validity(self):
        edges, nodes, node_info, cost = part.main("benchmarks/cm138a.txt")

        # Ensures that the count of the node_info structure is correct with respect to the number of nodes in each partition
        for partition in range(len(nodes)):
            for node in range(len(nodes[partition])):
                adjacent_nodes = []
                for i in range(len(nodes)):
                    adjacent_nodes.append(set())
                node_list_keys = list(nodes[partition].keys())
                for edge in range(len(nodes[partition][node_list_keys[node]][0])):
                    for adjacent_node in edges[nodes[partition][node_list_keys[node]][0][edge]][0]:
                        for i in range(len(nodes)):
                            if adjacent_node in nodes[i]:
                                adjacent_nodes[i].add(adjacent_node)
                for i in range(len(nodes)): 
                    self.assertTrue(node_info[node_list_keys[node]][i + 2] == len(adjacent_nodes[i]))
        print(cost)

if __name__ == '__main__':
    unittest.main()