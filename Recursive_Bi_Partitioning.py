import Partitioning as part
import math

def main(INPUT_TEST, RANDOM_SEED, NUM_PARTITIONS):
    num_cuts = math.log2(NUM_PARTITIONS)
    nodes_a, nodes_b, edges, cost = part.main_function(INPUT_TEST, RANDOM_SEED)
    num_blocks = len(nodes_a)
    print(edges)
    print("===========")
    print(nodes_b)
    nodes_a, nodes_b, edges = part.recursive_bi_partition(nodes_a, edges, num_blocks, RANDOM_SEED)
    print("nodes a")
    print(nodes_a)
    print("nodes b")
    print(nodes_b)
main("benchmarks/cm138a.txt", 3, 4)