from copy import deepcopy
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

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

def split_nodes(nodes):
    """Splits the nodes into 2 equally sized lists (within 1)

    Args:
        nodes (list): nodes of the circuit

    Returns:
        nodes_a, nodes_b (lists): list of nodes of the circuit
    """
    nodes_a = {}
    nodes_b = {}
    for i in range(len(nodes)):
        if (i % 2 == 0):
            nodes_a[i] = deepcopy(nodes[i])
        else:
            nodes_b[i] = deepcopy(nodes[i])
    return nodes_a, nodes_b


def split_nodes_random(nodes):
    """Splits the nodes randomly into 2 equally sized lists (within 1)

    Args:
        nodes (list): nodes of the circuit

    Returns:
        nodes_a, nodes_b (lists): list of nodes of the circuit
    """
    nodes_a = {}
    nodes_b = {}
    num_nodes = len(nodes)
    node_locations = [i for i in range(num_nodes)]
    random.shuffle(node_locations)
    for i in range(len(nodes)):
        if (node_locations[i] % 2 == 0):
            nodes_a[i] = deepcopy(nodes[i])
        else:
            nodes_b[i] = deepcopy(nodes[i])
    return nodes_a, nodes_b

def split_nodes_random_bi_partitioning(nodes):
    """Splits the nodes randomly into 2 equally sized lists (within 1)

    Args:
        nodes (list): nodes of the circuit

    Returns:
        nodes_a, nodes_b (lists): list of nodes of the circuit
    """
    nodes_a = {}
    nodes_b = {}
    num_nodes = len(nodes)
    node_locations = list(nodes.keys())
    counter = 0
    random.shuffle(node_locations)
    for i in node_locations:
        if (counter % 2 == 0):
            nodes_a[i] = deepcopy(nodes[i])
        else:
            nodes_b[i] = deepcopy(nodes[i])
        counter += 1
    return nodes_a, nodes_b

def get_cost(nodes_a, nodes_b, edges):
    """Gets the cost (number of cuts)

    Args:
        nodes_a (list): list of nodes in partition a
        nodes_b (list): list of nodes in partition b
        edges (list): list of edges

    Returns:
        cost: cost of the circuit in terms of number of partition cuts
    """
    for i in range(len(edges)):
        all_in_set_a = 0
        all_in_set_b = 0
        in_other_partition = 0
        nodes_in_a = 0
        for j in range(len(edges[i][0])):
            edge = edges[i][0][j]
            if (edge in nodes_a):
                nodes_in_a += 1
                all_in_set_a = 1
            elif (edge in nodes_b):
                all_in_set_b = 1
            else:
                in_other_partition += 1
        edges[i][2] = nodes_in_a
        if ((all_in_set_a and all_in_set_b) or (in_other_partition > 0 and in_other_partition < len(edges[i][0]))):
            edges[i][1] = 1
        else:
            edges[i][1] = 0
    return edges

def get_highest_cost(nodes):
    """Finds the highest gain in the node list

    Args:
        nodes (list): list of nodes, containing the associated nets, their gain, if they are locked or not,
                      and how many nodes are in the a partition

    Returns:
        highest_cost_node (int): the node with the highest cost
    """
    node_keys = list(nodes.keys())
    max_cost = -100000000
    highest_cost_node = -1
    for i in range(len(nodes)):
        if (nodes[node_keys[i]][2] == 1):
            continue
        if (nodes[node_keys[i]][1] > max_cost):
            max_cost = nodes[node_keys[i]][1]
            highest_cost_node = node_keys[i]
    return highest_cost_node

def get_highest_costs(nodes_a, nodes_b):
    high_cost_a = get_highest_cost(nodes_a)
    high_cost_b = get_highest_cost(nodes_b)
    return high_cost_a, high_cost_b

def calc_new_costs_vanilla(nodes_a, nodes_b, edges, swapped_node_1, swapped_node_2):
    delta_cost = 0
    edges_to_recalc = list(set(nodes_a[swapped_node_2][0] + nodes_b[swapped_node_1][0]))
    previous_cost = 0
    # for all of the edges
    for i in range(len(edges_to_recalc)):
        node_a_cost = 0
        node_b_cost = 0
        # for each node in the edge
        for j in range(len(edges[edges_to_recalc[i]][0])):
            if (edges[edges_to_recalc[i]][0][j] in nodes_a):
                node_a_cost = 1
            if (edges[edges_to_recalc[i]][0][j] in nodes_b):
                node_b_cost = 1
            if (node_a_cost & node_b_cost):
                break
        # if there are nodes on both sides
        if (node_a_cost & node_b_cost):
            if (edges[edges_to_recalc[i]][1] == 0):
                delta_cost += 1
            edges[edges_to_recalc[i]][1] = 1
        # if both nodes are on the same side
        else:
            if (edges[edges_to_recalc[i]][1] == 1):
                delta_cost -= 1
            edges[edges_to_recalc[i]][1] = 0
    return edges, delta_cost


def calc_new_costs_modified(nodes_a, nodes_b, edges, swapped_node_1, swapped_node_2):
    delta_cost = 0
    edges_to_recalc = list(set(nodes_a[swapped_node_2][0] + nodes_b[swapped_node_1][0]))
    previous_cost = 0
    for i in range(len(edges_to_recalc)):
        node_a_cost = 0
        node_b_cost = 0

        for j in range(len(edges[edges_to_recalc[i]][0])):
            if (edges[edges_to_recalc[i]][0][j] in nodes_a):
                node_a_cost += 1
            if (edges[edges_to_recalc[i]][0][j] in nodes_b):
                node_b_cost += 1
            if (node_a_cost & node_b_cost):
                break

        # if all of the nodes are in the correct side
        if (len(edges[edges_to_recalc[i]][0]) == node_a_cost or len(edges[edges_to_recalc[i]][0]) == node_b_cost):
            if (edges[edges_to_recalc[i]][1] == 1):
                delta_cost -= 1
            edges[edges_to_recalc[i]][1] = 0
        # if there is a single node on the wrong side of the partition,
        # and if both nodes are connected in the same net
        # effectively, by taking this move, we're not maximizing the best edge move we could be 
        elif ((len(edges[edges_to_recalc[i]][0]) - 1 == node_a_cost or len(edges[edges_to_recalc[i]][0]) - 1 == node_b_cost)
                # and if they are both in the same net
                and (swapped_node_1 in edges[edges_to_recalc[i]][0] and swapped_node_2 in edges[edges_to_recalc[i]][0])):
            delta_cost += 1
            if (edges[edges_to_recalc[i]][1] == 0):
                delta_cost += 1
            edges[edges_to_recalc[i]][1] = 1

        else:
            if (edges[edges_to_recalc[i]][1] == 0):
                delta_cost += 1
            edges[edges_to_recalc[i]][1] = 1
    return edges, delta_cost


def calc_total_cost(edges):
    cost = 0
    for i in range(len(edges)):
        cost += edges[i][1]
    return cost

def calc_each_gain_initial_vanilla(nodes_a, nodes_b, edges):
    """Calculates the gain for each of the nodes

    Args:
        nodes_a (list): 
        nodes_b (list): 
        edges (list): 

    Returns:
       nodes_a, nodes_b: the nodes with their updated gains
    """
    node_a_keys = list(nodes_a.keys())
    for i in range(len(nodes_a)):
        gain = 0
        for j in range(len(nodes_a[node_a_keys[i]][0])):
            if(nodes_a[node_a_keys[i]][2] == 1):
                continue
            if (edges[nodes_a[node_a_keys[i]][0][j]][1] == 1):
                gain += 3 / (len(edges[nodes_a[node_a_keys[i]][0][j]][0]))
            else:
                gain -= 1
        nodes_a[node_a_keys[i]][1] = gain
    node_b_keys = list(nodes_b.keys())
    for i in range(len(nodes_b)):
        gain = 0
        for j in range(len(nodes_b[node_b_keys[i]][0])):
            if(nodes_b[node_b_keys[i]][2] == 1):
                continue
            if (edges[nodes_b[node_b_keys[i]][0][j]][1] == 1):
                gain += 3 / (len(edges[nodes_b[node_b_keys[i]][0][j]][0]))
            else:
                gain -= 1
        nodes_b[node_b_keys[i]][1] = gain
    return nodes_a, nodes_b


def calc_each_gain_initial_vanilla2(nodes_a, nodes_b, edges, swapped_node_1, swapped_node_2):
    """Calculates the gain for each of the nodes, taking into consideration the correction cost

    Args:
        nodes_a (list): 
        nodes_b (list): 
        edges (list): 
        swapped_node_1 (int): key value of the highest value node in a
        swapped_node_2 (int): key value of the highest value node in b

    Returns:
       nodes_a, nodes_b: the nodes with their updated gains
    """
    node_a_keys = list(nodes_a.keys())
    for i in range(len(nodes_a)):
        gain = 0
        for j in range(len(nodes_a[node_a_keys[i]][0])):
            if(nodes_a[node_a_keys[i]][2] == 1):
                continue
            if (edges[nodes_a[node_a_keys[i]][0][j]][1] == 1):
                if ((swapped_node_1 in edges[nodes_a[node_a_keys[i]][0][j]][0] and swapped_node_2 in edges[nodes_a[node_a_keys[i]][0][j]][0])
                    and (edges[nodes_a[node_a_keys[i]][0][j]][2] == len(edges[nodes_a[node_a_keys[i]][0][j]][0]) - 1)):
                    gain += 1
                gain += 3 / (len(edges[nodes_a[node_a_keys[i]][0][j]][0]))
            else:
                gain -= 1
        nodes_a[node_a_keys[i]][1] = gain
    node_b_keys = list(nodes_b.keys())
    for i in range(len(nodes_b)):
        gain = 0
        for j in range(len(nodes_b[node_b_keys[i]][0])):
            if (edges[nodes_b[node_b_keys[i]][0][j]][1] == 1):
                if(nodes_b[node_b_keys[i]][2] == 1):
                    continue
                if ((swapped_node_1 in edges[nodes_b[node_b_keys[i]][0][j]][0] and swapped_node_2 in edges[nodes_b[node_b_keys[i]][0][j]][0])
                    and (edges[nodes_b[node_b_keys[i]][0][j]][2] == 1)):
                    gain += 1
                gain += 3 / (len(edges[nodes_b[node_b_keys[i]][0][j]][0]))
            else:
                gain -= 1
        nodes_b[node_b_keys[i]][1] = gain
    return nodes_a, nodes_b

def swap_nodes(nodes_a, nodes_b, edges, node_to_swap_a, node_to_swap_b):
    """Swaps the position of 2 nodes, and updates the amount of nodes in each partition

    Args:
        nodes_a (list): 
        nodes_b (list): 
        edges (list): 
        node_to_swap_a (int): the node to be swapped from nodes_a
        node_to_swap_b (int): the node to be swapped from nodes_b

    Returns:
        nodes_a, nodes_b: node lists with swaps
    """
    temp = deepcopy(nodes_b[node_to_swap_b])
    # for all of the edges that the node we are about to swap are associated to
    # increase the amount of a nodes that belong in that side
    for i in range(len(nodes_b[node_to_swap_b][0])):
        edges[nodes_b[node_to_swap_b][0][i]][2] += 1
    for i in range(len(nodes_a[node_to_swap_a][0])):
        edges[nodes_a[node_to_swap_a][0][i]][2] -= 1
    del nodes_b[node_to_swap_b]
    nodes_a[node_to_swap_b] = temp
    temp = deepcopy(nodes_a[node_to_swap_a])
    nodes_b[node_to_swap_a] = temp
    del nodes_a[node_to_swap_a]
    # the 1 is a "lock", indicating that this node has been swapped on this pass
    nodes_a[node_to_swap_b][2] = 1
    nodes_b[node_to_swap_a][2] = 1
    return nodes_a, nodes_b

def unlock_nodes(nodes):
    node_a_keys = list(nodes.keys())
    for i in range(len(nodes)):
        nodes[node_a_keys[i]][2] = 0
    return nodes

def unlock_all_nodes(nodes_a, nodes_b):
    nodes_a = unlock_nodes(nodes_a)
    nodes_b = unlock_nodes(nodes_b)
    return nodes_a, nodes_b

def get_values(netlist):
    """gets descriptive values of netlist

    Args:
        netlist (list): list version of netlist

    Returns:
        num_blocks (int): number of blocks in netlist
        num_connections (int): number of connections between cells
        num_rows (int): number of grid rows for circuit to be placed
        num_columns (int): number of grid columns for circuit to be placed
        new_netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                            will contain the cost of the given net
    """
    num_blocks, num_connections, num_rows, num_columns = netlist[0]
    netlist = netlist[1:]
    new_netlist = {}
    for i in range(num_connections):
        new_netlist[int(i)] = []
        new_netlist[int(i)].append(netlist[i][1:])
        # the 0 will represet that the node has not yet been moved (it is unlocked)
        new_netlist[int(i)].append(0)
        new_netlist[int(i)].append(0)
    return num_blocks, num_connections, num_rows, num_columns, new_netlist

def init_cell_placements(num_blocks, num_rows, num_connections, num_columns, netlist):
    """Places cells into the nxm grid as specified by the input file (random locations)

    Args:
        num_blocks (int): [description]
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
    for i in range(num_blocks):
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


def main_function(input_file, random_seed):
    # random.seed(9)
    random.seed(random_seed)
    edges = parse_netlist(input_file)
    num_blocks, num_connections, num_rows, num_columns, edges = get_values(edges)
    nodes = init_cell_placements(num_blocks, num_rows, num_connections, num_columns, edges)
    nodes_a, nodes_b = split_nodes_random(nodes)
    edges = get_cost(nodes_a, nodes_b, edges)

    # fig, ax1 = plt.subplots()
    fig = 0
    ax1 = 0
    nodes_a, nodes_b, edges = loop_3(nodes_a, nodes_b, edges, num_blocks, fig, ax1)
    # plt.pause(.5)

    new_edges = get_cost(nodes_a, nodes_b, edges)
    new_cost = calc_total_cost(new_edges)

    return nodes_a, nodes_b, edges, new_cost

def get_uncut_edges(edges):
    uncut_edges = []
    for i in range(len(edges)):
        if (edges[i][1] == 0):
            uncut_edges.append(i)
    return uncut_edges

def recursive_bi_partition(nodes, edges, num_blocks, seed):
    """Performs bi_partitioning

    Args:
        nodes (dict): dict of nodes: for each key [<list of associated edges>][<current gain>][<if the node is locked>]
        edges (dict): dict of edges: for each key [<list of nodes>][<1 or 0, representing cost>][<# nodes in partition a>]
    """
    print("Entering Recursive Bi-Partitioning")
    random.seed(seed)
    previous_uncut_edges = get_uncut_edges(edges)
    print("previous uncut edges: " + str(previous_uncut_edges))
    nodes_a, nodes_b = split_nodes_random_bi_partitioning(nodes)
    print("initial nodes a & b")
    print(nodes_a)
    print("===")
    print(nodes_b)
    edges = get_cost(nodes_a, nodes_b, edges)
    new_uncut_edges = get_uncut_edges(edges)
    print("new uncut edges: " + str(new_uncut_edges))
    print(edges)
    fig = 0
    ax1 = 0
    # print(nodes)
    nodes_a, nodes_b, edges = loop_3(nodes_a, nodes_b, edges, num_blocks, fig, ax1)
    edges = get_cost(nodes_a, nodes_b, edges)
    final_uncut_edges = get_uncut_edges(edges)
    print("final_uncut_edges: " + str(final_uncut_edges))

    return nodes_a, nodes_b, edges

def loop_3(nodes_a, nodes_b, edges, num_blocks, fig, ax1):
    """Main loop

    Args:
        nodes_a (dict): dict of nodes: for each key [<list of associated edges>][<current gain>][<if the node is locked>]
        nodes_b (dict): dict of nodes: for each key [<list of associated edges>][<current gain>][<if the node is locked>]
        edges (dict): dict of edges: for each key [<list of nodes>][<1 or 0, representing cost>][<# nodes in partition a>]
        num_blocks (int): number of nodes
        fig (graph): used for graphing
        ax1 (graph): used for graphing

    Returns:
        nodes_a, nodes_b, edges: partitioned nodes and edges
    """
    # plt.title("Cost vs. Number of Steps")
    # ax1.set_xlabel("Number of KL Iterations")
    # ax1.set_ylabel("Cost (Total Cuts)", c="red")
    cost_array = []
    i = 0
    best_cut_a = deepcopy(nodes_a)
    best_cut_b = deepcopy(nodes_b)
    best_edges = deepcopy(edges)
    while (i < 6):
        nodes_a, nodes_b = unlock_all_nodes(nodes_a, nodes_b)
        is_change = 0
        nodes_a, nodes_b = calc_each_gain_initial_vanilla(nodes_a, nodes_b, edges)
        cost = calc_total_cost(edges)
        temp_cost = cost

        for j in range(int(num_blocks / 2)):
            highest_cost_node_a, highest_cost_node_b = get_highest_costs(nodes_a, nodes_b)
            nodes_a, nodes_b = swap_nodes(nodes_a, nodes_b, edges, highest_cost_node_a, highest_cost_node_b)

            edges, delta_cost = calc_new_costs_vanilla(nodes_a, nodes_b, edges, highest_cost_node_a, highest_cost_node_b)
            nodes_a, nodes_b = calc_each_gain_initial_vanilla2(nodes_a, nodes_b, edges, highest_cost_node_a, highest_cost_node_b)
            temp_cost += delta_cost

            if (cost > temp_cost):
                cost = temp_cost
                best_cut_a = deepcopy(nodes_a)
                best_cut_b = deepcopy(nodes_b)
                best_edges = deepcopy(edges)
                is_change = 1

        # cost_array.append(cost)
        # ax1.scatter(i, cost, c="Red")
        # ax1.plot(range(i+1), cost_array, c="red", linestyle='-')
        # plt.pause(0.05)
        i = i + 1
        print(cost)
        if(is_change == 1):
            nodes_a = deepcopy(best_cut_a)
            nodes_b = deepcopy(best_cut_b)
            edges = deepcopy(best_edges)
        else:
            return best_cut_a, best_cut_b, best_edges
        
    return nodes_a, nodes_b, edges
    