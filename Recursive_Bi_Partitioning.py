from copy import deepcopy
import Partitioning as part
import math

def print_partitions(partitions):
    for i in range(len(partitions)):
        print("Partition " + str(i) + ":")
        print(partitions[i])

def get_cost_bi_recursive(partition_list, edges):
    for i in range(len(edges)):
        edges[i][1] = 0
    for i in range(len(edges)):
        this_edge = [0] * len(partition_list)
        for j in range(len(edges[i][0])):
            break_outer = 0
            for k in range(len(partition_list)):
                if edges[i][0][j] in partition_list[k]:
                    this_edge[k] += 1
        if (len(edges[i][0]) not in this_edge):
            edges[i][1] = 1

def main(INPUT_TEST, RANDOM_SEED, NUM_PARTITIONS):
    num_cuts = int(math.log2(NUM_PARTITIONS))
    nodes_a, nodes_b, edges, cost = part.main_function(INPUT_TEST, RANDOM_SEED)
    num_blocks = len(nodes_a)
    num_blocks_original = num_blocks
    partition_lists = [nodes_a, nodes_b]
    for _ in range(num_cuts - 1):
        temp_nodes = []
        while (partition_lists):
            nodes_c, nodes_d, edges = part.recursive_bi_partition(partition_lists.pop(0), edges, num_blocks, RANDOM_SEED)
            temp_nodes.append(nodes_c)
            temp_nodes.append(nodes_d)
        partition_lists = deepcopy(temp_nodes)
        num_blocks /= 2
    return partition_lists, edges, num_blocks_original
main("benchmarks/cm138a.txt", 3, 8)