#!/opt/python-3.4/linux/bin/python3

import re
import sys
import warnings

import numpy as np
import scipy as sp

from functools import reduce
from itertools import combinations

from scipy.sparse import (csc_matrix, csr_matrix, coo_matrix, dok_matrix)

np.set_printoptions(precision=3)

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

    # ratings_input = list(parsed_input_generator(sys.stdin))
except InputError as e:
    # there was an InputError so record the error and exit
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

# user_ids are the rows
sequence_user_ids   = [input[0] for input in ratings_input]
# movie_ids are the cols
sequence_movie_ids  = [input[1] for input in ratings_input]
# ratings are the values
sequence_ratings    = [input[2] for input in ratings_input]

# used to construct the matrices
unique_user_ids = np.unique(sequence_user_ids)
unique_movie_ids = np.unique(sequence_movie_ids)

size_user_id = max(unique_user_ids) + 1
size_movie_id = max(unique_movie_ids) + 1

ratings_arr = np.zeros(shape=(size_user_id, size_movie_id))
ratings_arr[sequence_user_ids, sequence_movie_ids] = sequence_ratings

co_occurance_arr = np.zeros(shape=(size_movie_id, size_movie_id))

for user_ratings in ratings_arr:
    movies_rated_by_user = np.flatnonzero(user_ratings)
    co_occurance_pairs = combinations(movies_rated_by_user, 2)
    for movie_id_1, movie_id_2 in co_occurance_pairs:
        co_occurance_arr[movie_id_1, movie_id_2] += 1
        co_occurance_arr[movie_id_2, movie_id_1] += 1
    for movie_id in movies_rated_by_user:
        co_occurance_arr[movie_id, movie_id] += 1

recommendation_arr = co_occurance_arr.dot(ratings_arr.T)
unseen_recs_arr = np.where(ratings_arr == 0, recommendation_arr.T, 0)

''' enumerates each row of user recommendations to get the user id
Then the user recommendations are enumerated so that each recommendation value
contains the movie_id as well (the index is the movie id). Then recommendations
with zero values are filtered out. Then the recommendations are sorted by the
recommendation value. Then we print
'''
print('='*80)
print('CO-OCCURANCE RECOMMENDATIONS')
print('='*80)
for id, user_recommendations in enumerate(unseen_recs_arr):
    if not user_recommendations.any():
        continue
    recs_with_movie_id = enumerate(user_recommendations)
    filtered_recs = filter(lambda i: i[1] != 0, recs_with_movie_id)
    sorted_recs = sorted(filtered_recs, key=lambda i: i[1], reverse=1)
    if len(sorted_recs):
        print('user', id, sorted_recs, '=> recommend', sorted_recs[0][0])
    else:
        print('user', id, '=> no recommendation')
print('='*80, '\n', '='*80, sep='')

def compute_similarity(u, v):
    non_zero_indices = np.logical_and(u>0, v>0)
    if not len(non_zero_indices):
        return -1
    u = u[non_zero_indices]
    v = v[non_zero_indices]
    return u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))

user_similarity_arr = np.zeros(shape=(size_user_id, size_user_id))

user_id_pairs = combinations(unique_user_ids, 2)

for user_id_1, user_id_2 in user_id_pairs:
    similarity = compute_similarity(
        ratings_arr[user_id_1],
        ratings_arr[user_id_2]
    )
    user_similarity_arr[user_id_1, user_id_2] = similarity
    user_similarity_arr[user_id_2, user_id_1] = similarity

weighted = ratings_arr.T.dot(user_similarity_arr).T
counts_with_self = np.tile(co_occurance_arr.diagonal(), (size_user_id, 1))
rated_mask = (ratings_arr > 0).astype(int)
counts = counts_with_self - rated_mask
recommendations = np.divide(weighted, counts)

recommendations[np.isnan(recommendations)] = 0
recommendations[rated_mask == 1] = 0

print()

print('='*80)
print('USER-SIMILARITY RECOMMENDATIONS')
print('='*80)
for id, user_recommendations in enumerate(recommendations):
    if not user_recommendations.any():
        continue
    recs_with_movie_id = enumerate(user_recommendations)
    filtered_recs = filter(lambda i: i[1] != 0, recs_with_movie_id)
    sorted_recs = sorted(filtered_recs, key=lambda i: i[1], reverse=1)
    if len(sorted_recs):
        print('user', id, sorted_recs, '=> recommend', sorted_recs[0][0])
    else:
        print('user', id, '=> no recommendation')
print('='*80, '\n', '='*80, sep='')

