from copy import deepcopy
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

BALANCING_FACTOR = 1.03
NUM_PARTITIONS = 2

# Parameters defined for balancing partition sizes
global W_MIN 
global W_MAX 
def parse_netlist(rel_path):
    """Parses the netlist

    Args:
        rel_path (string): relative path to location of the input file

    Returns:
        list: list version of netlist
    """
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

    abs_file_path = os.path.join(script_dir, rel_path)
    # reading in the file
    data = []
    with open(abs_file_path) as rfile:
        data_raw = rfile.readlines()
        for line in data_raw:
            data.append(line.strip())
    data = list(filter(('').__ne__, data))
    split_data = []
    for i in range(int(len(data))):
        temp = (data[i].split())
        temp = list(map(int, temp))
        split_data.append(temp)
    return split_data

def vertex_info_init(nodes, num_nodes, edges):
    vertex_info = {}

    # finds which partition each vertex of each edge belongs to
    which_partition = {}
    for i in range(len(edges)):
        which_partition[i] = []
        for j in range(len(edges[i][0])):
            for k in range(len(nodes)):
                if (edges[i][0][j] in nodes[k]):
                    which_partition[i].append(k)
    # print("which partition")
    # print(which_partition)
    for i in range(num_nodes):
        # get the associated vertices
        vertex_info[i] = []
        # current partition of the node
        vertex_info[i].append(-1)
        # the partitions that the node attaches to (N)
        vertex_info[i].append(set())
        # the partitions that the node attaches to externally AND internally (ID & ED)
        for j in range(len(nodes)):
            # vertex_info[i].append(0)
            vertex_info[i].append(set())
        for j in range(len(nodes)):
            # if the node is in the partition
            if (i in nodes[j]):
                vertex_info[i][0] = j
                # look at all of the edges of the node:
                for k in range(len(nodes[j][i][0])):
                    # for each edge, look at each associated vertex
                    for l in range(len(edges[nodes[j][i][0][k]][0])):
                        vertex_info[i][1].add(which_partition[nodes[j][i][0][k]][l])
                        # 1st 2 indicies are the partition of the node, and the partitions the node attaches to
                        # this assigns the node to the ED field
                        # This line of code below works just fine, but it might not be needed. Will be counting
                        # the number of nodes in a partition instead of storing each one for simplicity, unless
                        # that backfires.
                        # Remember to change the earlier line in the first j loop to use sets instead if we choose
                        # to use this bad boi.
                        vertex_info[i][which_partition[nodes[j][i][0][k]][l] + 2].add(edges[nodes[j][i][0][k]][0][l])
                        # vertex_info[i][which_partition[nodes[j][i][0][k]][l] + 2] += 1
                for k in range(2, len(vertex_info[i])):
                    vertex_info[i][k] = len(vertex_info[i][k])
                break
        # lock condition
        vertex_info[i].append(0)
    # print("vertex_info")
    # print(vertex_info)
    return vertex_info

def get_values(netlist):
    """gets descriptive values of netlist

    Args:
        netlist (list): list version of netlist

    Returns:
        num_nodes (int): number of blocks in netlist
        num_connections (int): number of connections between cells
        num_rows (int): number of grid rows for circuit to be placed
        num_columns (int): number of grid columns for circuit to be placed
        new_netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                            will contain the cost of the given net
    """
    num_nodes, num_connections, num_rows, num_columns = netlist[0]
    netlist = netlist[1:]
    new_netlist = {}
    for i in range(num_connections):
        new_netlist[int(i)] = []
        new_netlist[int(i)].append(netlist[i][1:])
        # the 0 will represet that the node has not yet been moved (it is unlocked)
        new_netlist[int(i)].append(0)
        new_netlist[int(i)].append(0)
    return num_nodes, num_connections, num_rows, num_columns, new_netlist

def init_cell_placements(num_nodes, num_rows, num_connections, num_columns, netlist):
    """Places cells into the nxm grid as specified by the input file (random locations)

    Args:
        num_nodes (int): [description]
        num_rows (int): number of rows in cell grid
        num_connections (int): number of nets in cell grid
        num_columns (int): number of columns in cell grid
        netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                        will contain the cost of the given net (left for init_cell_placement)

    Returns:
        dict: contains the locations of each block in the format of:
              block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    """
    block_locations = {}
    avail_locations = []
    for i in range(num_rows):
        for j in range(num_columns):
            temp_loc = [i, j]
            avail_locations.append(temp_loc)
    random.shuffle(avail_locations)

    # after this, block locations looks like: block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    for i in range(num_nodes):
        block_locations[int(i)] = []
        associated_nets = []
        for j in range(num_connections):
            if int(i) in netlist[j][0]:
                associated_nets.append(j)
        block_locations[int(i)].append(associated_nets)
        block_locations[int(i)].append(0)
        block_locations[int(i)].append(0)
    avail_locations = []
    # fill in blank block locations
    return block_locations

def split_nodes_random(nodes, num_partitions):
    """Splits the nodes randomly into 2 equally sized lists (within 1)

    Args:
        nodes (list): nodes of the circuit

    Returns:
        nodes_a, nodes_b (lists): list of nodes of the circuit
    """
    new_nodes = []
    for _ in range(num_partitions):
        new_nodes.append({})
    num_nodes = len(nodes)
    node_locations = [i for i in range(num_nodes)]
    random.shuffle(node_locations)
    for i in range(len(nodes)):
        new_nodes[node_locations[i] % num_partitions][i] = nodes[i]
    # print_node_lists(new_nodes)
    return new_nodes

def print_node_lists(nodes):
    num_node_lists = len(nodes)
    for i in range(num_node_lists):
        print("Node list number " + str(i) + ":")
        print(nodes[i])

def block_swap(nodes, node_info, edges, node_location_1, node_to_swap_1, node_location_2):
    """Swaps 2 blocks

    Args:
        nodes (list): list of dicts containing the variable number of partitions: 
                      
        node_info (dict): [partition number][key (node number)][partition of vertex][<partitions attached to>][<number of attached nodes in partition A>]...
                          [<locked status>]  
        edges (dic): for each key [<list of nodes>][<1 or 0, representing cost>][<# nodes in partition a>]
        node_location_1 (int): the partition number that the node is located in
        node_to_swap_1 (int): the node to swap
        node_location_2 (int): the partition number that the node is located in
        node_to_swap_2 (int): the node to swap

    Returns:
        null (modified node_info, swapped nodes)
    """

    # modifies the number of nodes in each partition
    # I don't know if this is right, but it seems to be working
    counted_vertices = set()
    for i in range(len(nodes[node_location_1][node_to_swap_1][0])):
        for j in range(len(edges[nodes[node_location_1][node_to_swap_1][0][i]][0])):
            if edges[nodes[node_location_1][node_to_swap_1][0][i]][0][j] in counted_vertices:
                continue
            counted_vertices.add((edges[nodes[node_location_1][node_to_swap_1][0][i]][0][j]))
            node_info[edges[nodes[node_location_1][node_to_swap_1][0][i]][0][j]][node_location_2 + 2] += 1
            node_info[edges[nodes[node_location_1][node_to_swap_1][0][i]][0][j]][node_location_1 + 2] -= 1
    temp = deepcopy(nodes[node_location_1][node_to_swap_1])
    del nodes[node_location_1][node_to_swap_1]
    nodes[node_location_2][node_to_swap_1] = temp
    # will also need to update the vertex_info[i][1] block, to see if there are other nodes outside of the 
    # partition it connects to, but I don't know if we need this field. Will keep it in for now.
    for i in range(len(node_info)):
        for j in range(2, len(node_info[i]) - 1):
            if not node_info[i][j]:
                node_info[i][1].discard(j - 2)
            else:
                node_info[i][1].add(j - 2)

    # swaps the partition location of the given nodes in node_info
    node_info[node_to_swap_1][0] = node_location_2

    # update the lock
    node_info[node_to_swap_1][-1] = 1
    return

def calc_cost(edges, nodes):
    cost = 0
    for i in range(len(edges)):
        cut_edge = -1
        for j in range(len(edges[i][0])):
            this_cut_edge = -1
            for k in range(len(nodes)):
                if edges[i][0][j] in nodes[k]:
                    this_cut_edge = k
                    break
            if (cut_edge is not -1 and this_cut_edge is not cut_edge):
                cost += 1
                break
            else:
                cut_edge = this_cut_edge
    return cost
def calc_cost_cuts(node_info):
    """Sums the total edge cuts 

    Args:
        node_info (dict): [partition number][key (node number)][partition of vertex][<partitions attached to>][<number of attached nodes in partition A>]...
                          [<locked status>]  

    Returns:
        cost: number of edge cuts
    """
    cost = 0
    for i in range(len(node_info)):
        edge_cuts = 0
        for j in range(2, len(node_info[i]) - 1):
            if(node_info[i][j] > 0): 
                edge_cuts += 1
            if (edge_cuts > 1):
                cost += 1
                break
    return cost

def calc_gain(nodes, node_info, node, edges):
    """Calculates the gain of a node

    Args:
        nodes (_type_): _description_
        node_info (_type_): _description_
        node (int): the node of interest

    Returns:
        null
    """
    # print(node)
    # if (node == 15):
    #     print("edges")
    #     print(edges)
    #     print("initial node list")
    #     print_node_lists(nodes)
    #     print("initial node info")
    #     print(node_info)
    i_deg = node_info[node][node_info[node][0] + 2]
    e_deg = -1
    max_partition = -1
    is_solved = 0
    for i in range(2, len(node_info[node]) - 1):
        # Skip the internal node
        if (i - 2 == node):
            continue
        if (node_info[node][i] > e_deg):
            e_deg = node_info[node][i]
            max_partition = i - 2
        if (node_info[node][i] > 0):
            is_solved = 1
    # Because it's a greedy algorithm, if all of the nodes are already in a single partition, we don't need to swap.
    # We only will be considering boundary nodes.
    if (is_solved == 0):
        return
    
    if (len(nodes[max_partition]) + 1 <= W_MAX and len(nodes[node_info[node][0]]) - 1 >= W_MIN):
        print("Node in question: " + str(node) + ", Max external degree: " + str(e_deg) + ", Partition External: " + str(max_partition) + ", Internal Degree: " + str(i_deg) + ", Internal partition: " + str(node_info[node][0]))
        # Only make the swap if we're in the acceptable imbalance parameters
        print("Number of nodes (B): " + str(len(nodes[max_partition])) + ", W_MAX: " + str(W_MAX) + ", Number of nodes (A): " + str(len(nodes[node_info[node][0]])) + ", W_MIN: " + str(W_MIN))
        print_node_lists(nodes)
        print("node_info")
        print(node_info)
        print("edges")
        print(edges)
        print(node_info)
        # We make the swap if the degree is higher, or if there's an imbalance
        if (e_deg > i_deg or (e_deg == i_deg and len(nodes[max_partition]) - len(nodes[node_info[node][0]]) > 0)):
            print(node)
            block_swap(nodes, node_info, edges, node_info[node][0], node, max_partition)
    return

def unlock_nodes(node_info):
    for i in range(len(node_info)):
        node_info[i][-1] = 0


def main(input_file):
    # random.seed(9)
    edges = parse_netlist(input_file)
    num_nodes, num_connections, num_rows, num_columns, edges = get_values(edges)

    # Set the global constants, based on the size of the inputs
    global W_MIN 
    W_MIN = int(.9 * (num_nodes / NUM_PARTITIONS))
    global W_MAX 
    W_MAX = math.ceil(BALANCING_FACTOR * (num_nodes / NUM_PARTITIONS))

    nodes = init_cell_placements(num_nodes, num_rows, num_connections, num_columns, edges)
    nodes = split_nodes_random(nodes, NUM_PARTITIONS)
    node_info = vertex_info_init(nodes, num_nodes, edges)
    print(calc_cost(edges, nodes))
    # print(node_info)
    for i in range(8):
        vertices = list(range(len(node_info)))
        random.shuffle(vertices)
        print("iteration number " + str(i))
        for i in range(num_nodes):
            calc_gain(nodes, node_info, vertices[i], edges)
        unlock_nodes(node_info)
    # print(calc_cost(node_info))
    print("final node info")
    print(node_info)
    print("final set of nodes")
    print_node_lists(nodes)
    # print(edges)
    # print("first node info")
    # print(node_info)
    # block_swap(nodes, node_info, edges, 0, 1, 1)
    # print("second node info")
    # print(node_info)
    print("edges")
    print(edges)
    # print_node_lists(nodes)
    return edges, nodes, node_info, calc_cost(edges, nodes)

# main()