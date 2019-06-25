import os
import random
import threading
import numpy as np
from math import expm1, e
from sklearn.metrics.pairwise import cosine_similarity

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


def calculate_random_p_i_u_v(graph, item):
    p_i_u_v = {}
    for node1 in graph.keys():
        if node1 not in p_i_u_v:
            p_i_u_v[node1] = {}
        for node2 in graph[node1]:
            p_i_u_v[node1][node2] = random.uniform(0.001, 0.2)
    return p_i_u_v


def get_p_i_u_v(i, u, v, ratings):
    ratings_i = ratings[np.where(ratings[:,1] == i)].copy()

    # print(ratings_i)

    if u in ratings_i[:, 0]:
        idx_u = np.where(ratings_i[:,0] == u)
    else:
        return float(0.)

    if v in ratings_i[:, 0]:
        idx_v = np.where(ratings_i[:,0] == v)
    else:
        return float(0.)

    t_u = ratings_i[idx_u, 5]
    t_v = ratings_i[idx_v, 5]
    if t_u >= t_v:
        return float(0.)

    N_u = number_of_ratings_per_user[u]
    N_slash_u = 0
    for neighbor in graph[u]:
        N_slash_u += number_of_ratings_per_user[neighbor]
    if N_slash_u == 0:
        N_slash_u = np.nextafter(0, 1)

    ratings_of_user_u = ratings[np.where(ratings[:,0] == u)].copy()
    ratings_of_user_v = ratings[np.where(ratings[:,0] == v)].copy()
    common_ratings_u = []
    common_ratings_v = []
    for productID in ratings_of_user_u[:,1]:
        if productID in ratings_of_user_v[:,1]:
            common_ratings_u.append(ratings_of_user_u[np.where(ratings_of_user_u[:,1] == productID), 3].item())
            common_ratings_v.append(ratings_of_user_v[np.where(ratings_of_user_v[:,1] == productID), 3].item())
    common_ratings_u = np.array(common_ratings_u)
    common_ratings_v = np.array(common_ratings_v)
    s_u_v = cosine_similarity(common_ratings_u.reshape(1, -1), common_ratings_v.reshape(1, -1)).item()

    p = expm1((-abs(t_u-t_v))/a) + 1
    p *= (expm1(s_u_v))/(e-1)
    p *= N_u/N_slash_u
    p *= (expm1(-1) - expm1(-s_u_v)) / (expm1(-1))

    return p


def calculate_p_i_u_v(graph, ratings, item):
    p_i_u_v = {}
    for node1 in graph.keys():
        if node1 not in p_i_u_v:
            p_i_u_v[node1] = {}
        for node2 in graph[node1]:
            p_i_u_v[node1][node2] = get_p_i_u_v(item, node1, node2, ratings)
    return p_i_u_v


def build_ratings_array(filepath):
    ratings = np.loadtxt(filepath)
    ratings = ratings.astype('int32')
    userID, number_of_ratings = np.unique(ratings[:,0], return_counts=True)
    number_of_ratings_per_user = dict(zip(userID, number_of_ratings))
    a = np.max(ratings[:,5]) - np.min(ratings[:,5])
    return ratings, number_of_ratings_per_user, a


def write_r_i_u_sizes_in_file(R_i_u, filepath):
    if os.path.exists(filepath):
        f = open(filepath, "w")
    else:
        f = open(filepath, "x")

    for item in R_i_u.keys():
        for node in R_i_u[item].keys():
            f.write(item + ' ' + node + ' ' + len(R_i_u[item][node]) + '\n')
    f.close()


def subgraph(graph, size):
    random.seed(43)
    subgraph = {}

    source = random.randint(1, 18089)

    # Mark all the vertices as not visited
    visited = [False] * (len(graph.keys()))

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append(source)
    subgraph[source] = []
    current_size = 1

    while queue:

        # Dequeue a vertex from
        # queue
        s = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s. If a adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for i in graph[source]:
            if i not in subgraph.keys():
                queue.append(i)
                subgraph[i] = []
                current_size += 1

            if current_size >= size:
                break

        if current_size >= size:
            break

    for node in subgraph.keys():
        possible_neighbors = graph[node]
        for neighbor in possible_neighbors:
            if neighbor in subgraph.keys():
                subgraph[node].append(neighbor)

    return subgraph


def thread_function(node, graph, p_i_u_v):
    R_i_u = find_reachable_set(graph, p_i_u_v, node, probability_threshold=0.5, number_of_samples=3)
    print('R_i_u size', len(R_i_u))
    f.write(str(item) + ' ' + str(node) + ' ' + str(len(R_i_u)) + '\n')


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


graph_filepath = 'C:\\Users\\gkikas\\Documents\\Datasets\\Social Networks\\Epinions-JiliangTang-MichiganUniversity\\epinion_with_rating_timestamp_txt\\trust.txt'
ratings_filepath = 'C:\\Users\\gkikas\\Documents\\Datasets\\Social Networks\\Epinions-JiliangTang-MichiganUniversity\\epinion_with_rating_timestamp_txt\\rating_with_timestamp.txt'

graph = create_social_network(graph_filepath)

graph = subgraph(graph, 20)
# write_graph_in_file(graph, "mygraph.txt")
print(graph)
# create 200 random items
items = list(range(1, 296278))
random.seed(42)
random.shuffle(items)
items = items[:200]

ratings, number_of_ratings_per_user, a = build_ratings_array(ratings_filepath)
ratings = ratings[np.where(ratings[:,1] < 200)].copy()
print("nodes", len(graph.keys()))

r_i_u_filepath = "r_i_u.txt"
if os.path.exists(r_i_u_filepath):
    f = open(r_i_u_filepath, "w")
else:
    f = open(r_i_u_filepath, "x")

# for item in items:
#     print(item)
#     p_i_u_v = calculate_p_i_u_v(graph, item)
#     number_of_nodes = 0
#     for node in graph.keys():
#         number_of_nodes += 1
#         print(number_of_nodes, "/", len(graph.keys()))
#         R_i_u = find_reachable_set(graph, p_i_u_v, node, probability_threshold=0.5, number_of_samples=3)
#         print('R_i_u size', len(R_i_u))
#         f.write(str(item) + ' ' + str(node) + ' ' + str(len(R_i_u)) + '\n')
# f.close()


for item in items:
    print(item)
    p_i_u_v = calculate_p_i_u_v(graph, ratings, item)
    number_of_nodes = 0
    for nodes in grouped(graph.keys(), 8):

        number_of_nodes += 8
        print(number_of_nodes, "/ 18089")

        threads = list()
        for index in range(8):
            x = threading.Thread(target=thread_function, args=(nodes[index], graph, p_i_u_v))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()


f.close()

