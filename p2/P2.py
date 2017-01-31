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

from scipy.sparse import (csc_matrix, csr_matrix, coo_matrix, dok_matrix)

np.set_printoptions(precision=3)

INPUT_RE_STR = r'^(?:\s*#.*)|^(?:\s*(?P<a>[A-Za-z]+)\s*,\s*(?P<b>[A-Za-z]+)' \
    '\s*(?:#.*)?)$'

INPUT_RE = re.compile(INPUT_RE_STR)

# scipy throws warnings which we do not want to see
warnings.filterwarnings('ignore')


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def parsed_input_generator(input_lines):
    for input_line in input_lines:
        match = INPUT_RE.match(input_line)

        # there was no match at all, this line is not valid!
        if match is None:
            raise InputError('Invalid input `%s`' % input_line.rstrip('\n'))

        groups = match.groups()

        # this line starts with a comment so we will skip it
        if None in groups:
            continue

        # unpack the groups, which is a 3-tuple of matched strings
        a_str, b_str = groups

        # have the generator yield a 3-tuple of the parsed values
        yield a_str, b_str


try:
    # for every input line, parse the line and put the values into ratings
    f = open('sample_input.csv', 'r')
    graph_input = list(parsed_input_generator(f))
    f.close()
    #ratings_input = list(parsed_input_generator(sys.stdin))
except InputError as e:
    # there was an InputError so record the error and exit
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

input_lower = [list(map(lambda s: s.lower(), t)) for t in graph_input]
input_sorted = [sorted(l) for l in input_lower]
