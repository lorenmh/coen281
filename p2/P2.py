#!/opt/python-3.4/linux/bin/python3

'''
LAUREN HOWARD - COEN 281 - P1 1/22/2017
'''

import re
import sys
import warnings

import numpy as np
import scipy as sp

from functools import reduce
from itertools import combinations

np.set_printoptions(precision=3)

INPUT_RE_STR = r'^(?:\s*#.*)|^(?:\s*(?P<a>[A-Za-z]+)\s*,\s*(?P<b>[A-Za-z]+)' \
    '\s*(?:#.*)?)$'

INPUT_RE = re.compile(INPUT_RE_STR)


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def parsed_input_generator(input_lines):
    for input_line in input_lines:
        match = INPUT_RE.match(input_line)
        if match is None:
            raise InputError('Invalid input `%s`' % input_line.rstrip('\n'))
        groups = match.groups()
        # groups will only contain None if the input was a comment
        if None in groups:
            continue
        yield groups


try:
    # parse the input
    # f = open('sample_input.csv', 'r')
    # graph_input = list(parsed_input_generator(f))
    # f.close()
    graph_input = list(parsed_input_generator(sys.stdin))
except InputError as e:
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

# graph is represented as a dictionary. The keys are node_ids, the values are
# lists of node_ids which that node is connected to. For example,
# graph['A'] == ['B','C'] would mean that node A is connected to nodes B and C
graph = {}
for a, b in graph_input:
    a_edges = graph.get(a, set([]))
    b_edges = graph.get(b, set([]))

    a_edges.add(b)
    b_edges.add(a)

    graph[a] = a_edges
    graph[b] = b_edges


def ekey(edge):
    return tuple(sorted(edge))


edges_list = [ekey(edge) for edge in graph_input]
edges = set(edges_list)

if len(edges_list) != len(edges):
    print('Error with input: Duplicate edges were entered', file=sys.stderr)
    sys.exit(1)


def gna(graph, root_node_id):
    '''Girvan-Newman Algorithm for determining betweenness of edges.
    I have combined a number of steps into one single BFS. In the algorithm
    outlined in the book, there are 3 BFS performed.  In my BFS, I set the
    parents for each node. In the same BFS I determine whether a node is a leaf
    or not. I also group each level so when I begin calculating the betweenness
    I can walk from the deepest leaves and move up'''

    parent_nodes = {root_node_id: [None]}

    child_nodes = {}
    leaves = set([])
    levels = []
    node_weights = {}
    edge_weights = {}

    already_visited = set([])
    current_level = set([root_node_id])
    next_level = set([])

    while True:
        for parent_node_id in current_level:
            connected_nodes = graph[parent_node_id]

            children = (
                set(connected_nodes) - already_visited - current_level
            )

            child_nodes[parent_node_id] = children

            if not len(children):
                leaves.add(parent_node_id)
                continue

            for child_node_id in children:

                parents = parent_nodes.get(child_node_id, [])
                parents.append(parent_node_id)
                parent_nodes[child_node_id] = parents

                next_level.add(child_node_id)

        already_visited |= current_level
        levels.append(current_level)

        if len(next_level):
            current_level = next_level
            next_level = set([])
        else:
            break

    num_parents = {k: len(v) for k, v in parent_nodes.items()}

    #return parent_nodes, num_parents, levels, child_nodes, weights
    # now that the 'tree' has been constructed, let's assign points
    for level in reversed(levels):
        for node_id in level:
            if node_id in leaves:
                node_weights[node_id] = 1
                continue
            node_weight = 1
            for child_node_id in child_nodes[node_id]:
                edge_weight = (node_weights[child_node_id] /
                               num_parents[child_node_id])
                edge_weights[ekey((node_id, child_node_id))] = edge_weight
                node_weight += edge_weight
            node_weights[node_id] = node_weight
    return edge_weights


node_ids = set([i for sublist in graph_input for i in sublist])

weights_accumulator = {}
for root_node_id in node_ids:
    weights = gna(graph, root_node_id)
    for edge_key, weight in weights.items():
        edge_weight = weights_accumulator.get(edge_key, 0)
        edge_weight += weight
        weights_accumulator[edge_key] = edge_weight

for edge_key, weight in weights_accumulator.items():
    weights_accumulator[edge_key] = weight / 2


def cut(graph, edge_key):
    a, b = edge_key
    a_edges = graph[a]
    b_edges = graph[b]
    a_edges.remove(b)
    b_edges.remove(a)
    graph[a] = a_edges
    graph[b] = b_edges


def get_components(graph):
    components = []
    visited = set([])

    for node_id in node_ids:

        if node_id in visited:
            continue

        # bfs
        visited.add(node_id)
        current = set(graph[node_id])
        next = set([])
        component = [node_id]

        while True:
            for n in current:
                if n in visited:
                    continue
                visited.add(n)
                next |= (set(graph[n]) - visited)
                component.append(n)
            if len(next):
                current = next
                next = set([])
            else:
                break

        components.append(component)

    return sorted([sorted(component) for component in components])


def c2s(c):
    return ', '.join(map(lambda l: '(%s)' % ', '.join(l), c))


def print_components(components):
    print(len(components), 'cluster:', c2s(components))


w2e = {v: [] for k, v in weights_accumulator.items()}

for edge_key, weight in weights_accumulator.items():
    w2e[weight].append(edge_key)

edges_to_cut_grouped = [el[1] for el in sorted(w2e.items(), reverse=True)]

print_components(get_components(graph))

previous_components_len = 1
for edges_to_cut in edges_to_cut_grouped:
    for edge in edges_to_cut:
        cut(graph, edge)
    components = get_components(graph)
    l = len(components)
    if l != previous_components_len:
        previous_components_len = l
        print_components(get_components(graph))
