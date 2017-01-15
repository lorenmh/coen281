#!/opt/python-3.4/linux/bin/python3
import re
import sys
import warnings

import numpy as np
import scipy as sp

from itertools import permutations as permute

from scipy.sparse import csc_matrix as sparse_matrix

warnings.filterwarnings('ignore')
SPLIT_RE_STR = r',\s?'
SPLIT_RE = re.compile(SPLIT_RE_STR)

MAT_DTYPE = np.float64


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


def generate_user_occurance_matrix(movies_seen_by_user, size):
    user_occurance_matrix = sparse_matrix((size, size), dtype=MAT_DTYPE)
    permutations = permute(movies_seen_by_user, 2)
    for movie_1, movie_2 in permutations:
        user_occurance_matrix[movie_1, movie_2] = 1
        user_occurance_matrix[movie_2, movie_1] = 1
    for movie in movies_seen_by_user:
        user_occurance_matrix[movie, movie] = 1

    return user_occurance_matrix


def generate_user_ratings_matrix(movie_ratings_for_user, size):
    user_ratings_matrix = sparse_matrix((size, 1), dtype=MAT_DTYPE)
    for movie_id, rating in movie_ratings_for_user:
        user_ratings_matrix[movie_id, 0] = rating
    return user_ratings_matrix


# Filter out all empty lines
input_lines = filter(len, sys.stdin)

'''Ratings is a list of 3-tuples, (user_id, movie_id, rating)
'''
ratings = map(parse_input_line, input_lines)

# To get the max movie id
max_movie_id = max(ratings, key=lambda r: r[1])[1]
mat_size = max_movie_id + 1

# start with an empty co-occurance matrix
co_occurance_matrix = sparse_matrix((mat_size, mat_size), dtype=MAT_DTYPE)

ratings_grouped_by_user = reduce(
    group_by_user_reducer,
    ratings,
    {}
)

for user_id, user_ratings in ratings_grouped_by_user.iteritems():
    user_movies = map(lambda ur: ur[0], user_ratings)
    user_occurance_matrix = generate_user_occurance_matrix(
        user_movies, mat_size
    )
    co_occurance_matrix += user_occurance_matrix

for user_id, user_ratings in ratings_grouped_by_user.iteritems():
    user_ratings_matrix = generate_user_ratings_matrix(user_ratings, mat_size)
    recommendations_matrix = co_occurance_matrix * user_ratings_matrix
    all_recommendations = map(
        lambda i: (i[0][0], i[1]),
        recommendations_matrix.todok().items()
    )
    movies_already_seen = set(map(lambda ur: ur[0], user_ratings))
    unseen_recommendations = filter(
        lambda mr: mr[0] not in movies_already_seen,
        all_recommendations
    )
    unseen_recommendations.sort(key=lambda r: r[1], reverse=True)
    print (
        'user %d: %s => recommend %d' % (
            user_id,
            str(unseen_recommendations),
            unseen_recommendations[0][0]
        )
    )



# print co_occurance_matrix.toarray
