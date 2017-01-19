#!/opt/python-3.4/linux/bin/python3

import re
import sys
import warnings

import numpy as np
import scipy as sp

from functools import reduce
from itertools import permutations as permute

from scipy.sparse import (csc_matrix, csr_matrix, coo_matrix, dok_matrix)

INPUT_RE_STR = r'^(?:\s*#.*)|(?:\s*(?P<user_id>\d+)\s*,\s*(?P<movie_id>\d+)' \
    '\s*,\s*(?P<rating>\d+(?:\.\d+)?)\s*(?:#.*)?)$'
INPUT_RE = re.compile(INPUT_RE_STR)

# scipy throws warnings which we do not want to see
warnings.filterwarnings('ignore')


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def parsed_input_generator(input_lines):
    ''' parse_input_line will attempt to parse the user_id, movie_id, and
        rating from an input line. If there is an error with the input an
        InputError will be raised
    '''

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
        user_id_str, movie_id_str, rating_str = groups

        # because the regular expression matched I do not need to check for a
        # ValueError here
        user_id = int(user_id_str)
        movie_id = int(movie_id_str)
        rating = float(rating_str)

        # have the generator yield a 3-tuple of the parsed values
        yield user_id, movie_id, rating


try:
    # for every input line, parse the line and put the values into ratings
    f = open('sample_input.csv', 'r')
    ratings_input = list(parsed_input_generator(f))
    f.close()
except InputError as e:
    # there was an InputError so record the error and exit
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

user_ids =  [input[0] for input in ratings_input]
movie_ids = [input[1] for input in ratings_input]
ratings =   [input[2] for input in ratings_input]

size_user_id = max(user_ids) + 1
size_movie_id = max(movie_ids) + 1

# ratings_mat = csr_matrix((ratings, (movie_ids, user_ids)))
ratings_mat = np.zeros(shape=(size_movie_id, size_user_id))
ratings_mat[movie_ids, user_ids] = ratings

co_occurance_mat = np.zeros(shape=(size_movie_id, size_movie_id))

for user_ratings in ratings_mat.T:
    movies_rated_by_user = np.flatnonzero(user_ratings)
    co_occurance_pairs = permute(movies_rated_by_user, 2)
    user_co_occurance_mat = dok_matrix((size_movie_id, size_movie_id))
    for movie_id_1, movie_id_2 in co_occurance_pairs:
        user_co_occurance_mat[movie_id_1, movie_id_2] = 1
    for movie_id in movies_rated_by_user:
        user_co_occurance_mat[movie_id, movie_id] = 1
    co_occurance_mat += user_co_occurance_mat

recommendation_mat = co_occurance_mat * ratings_mat
