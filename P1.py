#!/opt/python-3.4/linux/bin/python3
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix as sparse_matrix
import re, sys

SPLIT_RE_STR = r',\s?'
SPLIT_RE = re.compile(SPLIT_RE_STR)

class InputError(Exception):
    def __init__(self, message):
        self.message = message

def parse_input_line(input_line):
    input_list = SPLIT_RE.split(input_line)
    if len(input_list) != 3:
        raise InputError('Invalid input line "%s"' % input_line)
    user_id_str, movie_id_str, rating_str = input_list
    try:
        user_id = int(user_id_str)
    except ValueError:
        raise InputError('Invalid user id "%s"' % user_id_str)
    try:
        movie_id = int(movie_id_str)
    except ValueError:
        raise InputError('Invalid movie id "%s"' % movie_id_str)
    try:
        rating = float(rating_str)
    except ValueError:
        raise InputError('Invalid rating "%s"' % rating_str)
    return user_id, movie_id, rating

def group_by_user_reducer(accumulator, user_movie_rating):
    user_id, movie_id, rating = user_movie_rating
    movie_ratings_for_user = accumulator.get(user_id, [])
    movie_ratings_for_user.append((movie_id, rating))
    accumulator[user_id] = movie_ratings_for_user
    return accumulator

# Filter out all empty lines
input_lines = filter(len, sys.stdin)

# Parse the CSV values from the lines of input
user_movie_ratings = map(parse_input_line, input_lines)

# Calculate the maximum movie id to create a sparse matrix
max_movie_id = max(user_movie_ratings, key=lambda umr: umr[1])[1]
# Create the sparse matrix
co_occurance_matrix = sparse_matrix(
        (max_movie_id, max_movie_id),
        dtype=np.uint64
)

movie_ratings_for_users = reduce(
        group_by_user_reducer,
        user_movie_ratings,
        {}
)

movie_ratings_for_users_sorted = {
        k:sorted(v) for (k,v) in movie_ratings_for_users.iteritems()}

print(movie_ratings_for_users_sorted)
