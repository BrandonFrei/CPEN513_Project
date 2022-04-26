import unittest
from copy import deepcopy
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


import K_Partitioning as part_k
import Partitioning as part
import Recursive_Bi_Partitioning as Re_Bi


global INPUT_TEST
INPUT_TEST = "benchmarks/cm138a.txt"
global RANDOM_SEED
RANDOM_SEED = 4
global NUMBER_OF_PARTITIONS
NUMBER_OF_PARTITIONS = 2

class TestStringMethods(unittest.TestCase):

    def test_output_validity_k(self):
        start = timer()
        edges, nodes, node_info, cost = part_k.main(INPUT_TEST, RANDOM_SEED, NUMBER_OF_PARTITIONS)
        end = timer()
        print("Time taken for k-way partitioning: " + str(end - start) + ". Cost: " + str(cost))
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
        # part_k.print_node_lists(nodes)

    def test_output_validity_bi(self):
        start = timer()
        partition_lists, edges, cost, num_blocks = Re_Bi.main(INPUT_TEST, RANDOM_SEED, NUMBER_OF_PARTITIONS)
        end = timer()
        print("Time taken for recursive bi-partitioning: " + str(end - start) + ". Cost: " + str(cost))
        all_nodes = []
        for i in range(len(partition_lists)):
            all_nodes = all_nodes + list(partition_lists[i].keys())
        all_nodes.sort()
        for i in range(num_blocks):
            self.assertTrue(i == all_nodes[i])
        # Re_Bi.print_partitions(partition_lists)
    # def test_output_validity(self):
    #     start = timer()
    #     nodes_a, nodes_b, edges, cost = part.main_function(INPUT_TEST, RANDOM_SEED)
    #     end = timer()
    #     print("time taken for recursive bi-partitioning: " + str(end - start) + ". Cost: " + str(cost))
    #     print(nodes_a)
    #     # print('========')
    #     print(edges)
    #     for i in range(len(edges)):
    #         cost = 0
    #         for j in range(len(edges[i][0])):
    #             if(edges[i][0][j] in nodes_a):
    #                 cost += 1
    #         correct_output = 0
    #         # if all the nodes are in nodes_a
    #         if (len(edges[i][0]) == cost):
    #             correct_output = 1
    #         # if all the nodes are in nodes_b
    #         elif (cost == 0):
    #             correct_output = 1
    #         # if there is a cut in an edge
    #         elif (edges[i][1] == 1 and len(edges[i][0]) != cost):
    #             correct_output = 1
    #         # the broken edge
    #         if (correct_output == 0):
    #             print(edges[i][0])
    #         self.assertTrue(correct_output == 1)

if __name__ == '__main__':
    unittest.main()