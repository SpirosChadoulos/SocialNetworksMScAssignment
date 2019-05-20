import os
import random

from dataset.reachable_set import find_reachable_set


def create_social_network(filepath):
    file = open(filepath, "r")
    graph = {}
    # j=0
    for line in file:
        # print(j)
        # j += 1
        edge = line.split()
        edge = [float(node) for node in edge]
        edge = [int(node) for node in edge]

        if edge[0] not in graph.keys():
            graph[edge[0]] = []
        if edge[1] not in graph.keys():
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])

    file.close()
    return graph


def write_graph_in_file(graph, filepath):
    if os.path.exists(filepath):
        f = open(filepath, "w")
    else:
        f = open(filepath, "x")

    for node in graph.keys():
        string_node = [str(node) + ": "]
        for x in graph[node]:
            string_node.append(str(x) + " ")
        string_node = ''.join(string_node)
        f.write(string_node + '\n')
    f.close()


def calculate_p_i(graph):
    p_i = {}
    for node1 in graph.keys():
        if node1 not in p_i:
            p_i[node1] = {}
        for node2 in graph[node1]:
            p_i[node1][node2] = random.random()
    return p_i


graph_filepath = 'C:\\Users\\gkikas\\Documents\\Datasets\\Social Networks\\Epinions-JiliangTang-MichiganUniversity\\epinion_with_rating_timestamp_txt\\trust.txt'
graph = create_social_network(graph_filepath)
# write_graph_in_file(graph, "mygraph.txt")
p_i = calculate_p_i(graph)

random_seed = random.choice(list(graph.keys()))
# random_seed = 20791
R_i = find_reachable_set(graph, p_i, random_seed, probability_threshold=0.5, number_of_samples=3)
print(random_seed)
print(R_i)