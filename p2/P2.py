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
    f = open('sample_input.csv', 'r')
    graph_input = list(parsed_input_generator(f))
    f.close()
    #ratings_input = list(parsed_input_generator(sys.stdin))
except InputError as e:
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

# graph is represented as a dictionary. The keys are node_ids, the values are
# lists of node_ids which that node is connected to. For example,
# graph['A'] == ['B','C'] would mean that node A is connected to nodes B and C
graph = {}
for a, b in graph_input:
    a_edges = graph.get(a, [])
    b_edges = graph.get(b, [])

    a_edges.append(b)
    b_edges.append(a)

    graph[a] = a_edges
    graph[b] = b_edges


def gna(root_node_id, graph):
    '''Girvan-Newman Algorithm for determining betweenness of edges.
    I have combined a number of steps into one single BFS. In the algorithm
    outlined in the book, there are 3 BFS performed.  In my BFS, I set the
    parents for each node. In the same BFS I determine whether a node is a leaf
    or not. I also group each level so when I begin calculating the betweenness
    I can walk from the deepest leaves and move up'''

    parent_nodes = {root_node_id: [None]}
    leaf_nodes = []
    levels = []

    already_visited = set([])
    current_level = set([root_node_id])
    next_level = set([])

    while True:
        for parent_node_id in current_level:
            connected_nodes = graph[parent_node_id]

            if not len(set(connected_nodes) - already_visited - current_level):
                leaf_nodes.append(parent_node_id)
                continue

            for child_node_id in connected_nodes:
                if child_node_id in (already_visited | current_level):
                    continue

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

    parent_lens = {k: len(v) for k, v in parent_nodes.items()}

    return parent_nodes, parent_lens, leaf_nodes, levels


node_ids = set([i for sublist in graph_input for i in sublist])
