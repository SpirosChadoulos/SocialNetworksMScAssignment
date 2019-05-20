import random


def reachable_set_deterministic_graph(seed, deterministic_graph):
    R_i_h = []

    queue = []

    queue.append(seed)
    R_i_h.append(seed)

    while queue:
        v = queue.pop(0)
        if v in deterministic_graph.keys():
            for neighbor in deterministic_graph[v]:
                if neighbor not in R_i_h:
                    queue.append(neighbor)
                    R_i_h.append(neighbor)

    R_i_h.remove(seed)
    return R_i_h


def find_reachable_set(graph, p_i, seed_id, probability_threshold=0.5, number_of_samples=13):
    c = {}
    for h in range(0, number_of_samples):
        deterministic_graph = {}
        for node1 in graph.keys():
            for node2 in graph[node1]:
                rnd = random.random()
                if node1 in p_i.keys():
                    if node2 in p_i[node1].keys():
                        if rnd <= p_i[node1][node2]:
                            if node1 not in deterministic_graph.keys():
                                deterministic_graph[node1] = []
                            deterministic_graph[node1].append(node2)
        R_i_h = reachable_set_deterministic_graph(seed_id, deterministic_graph)

        for j in R_i_h:
            if j not in c.keys():
                c[j] = 0
            c[j] = c[j] + 1

    R_i = []
    for node in c.keys():
        if c[node] >= probability_threshold*number_of_samples:
            R_i.append(node)
    return R_i
