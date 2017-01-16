#!/opt/python-3.4/linux/bin/python3
import re
import sys
import warnings

import numpy as np
import scipy as sp

from functools import reduce
from itertools import permutations as permute

from scipy.sparse import csc_matrix as sparse_matrix


# used to parse the input text of CSV values (which might have spaces with
# commas)
SPLIT_RE_STR = r',\s?'
SPLIT_RE = re.compile(SPLIT_RE_STR)

# any float type is acceptable here, I am using 64 bit because I am running
# this on a 64 bit machine.
MAT_DTYPE = np.float64


# scipy throws warnings which we do not want to see
warnings.filterwarnings('ignore')


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def parse_input_line(input_line):
    ''' parse_input_line will attempt to parse the user_id, movie_id, and
        rating from an input line. If there is an error with the input an
        InputError will be raised
    '''

    # remove the trailing newline character
    input_line = input_line.rstrip('\n')

    # use the regular expression ',\s?' to split the string by the comma
    # delimiter; "A,B,C" => ['A','B','C']
    input_list = SPLIT_RE.split(input_line)

    # if the input_list does not have length of 3 then there was an incorrectly
    # formatted input
    if len(input_list) != 3:
        raise InputError('Invalid input line `%s`' % input_line)

    # unpack the input_list
    user_id_str, movie_id_str, rating_str = input_list

    # attempt to convert user_id_str to an integer
    try:
        user_id = int(user_id_str)
    except ValueError:
        raise InputError(
            'Invalid user id `%s` in line `%s`' % (user_id_str, input_line)
        )

    # attempt to convert movie_id_str to an integer
    try:
        movie_id = int(movie_id_str)
    except ValueError:
        raise InputError(
            'Invalid movie id `%s` in line `%s`' % (movie_id_str, input_line)
        )

    # attempt to convert user_id_str to a float
    try:
        rating = float(rating_str)
    except ValueError:
        raise InputError(
            'Invalid rating `%s` in line `%s`' % (rating_str, input_line)
        )

    # we now have the three values which have the correct types so we return a
    # 3-tuple corresponding to these values.
    return (user_id, movie_id, rating)


def group_by_user_reducer(accumulator, user_movie_rating):
    ''' user_movie_rating should be a 3-tuple with the following format:
            (user_id, movie_id, rating)

        This reducer will group the movie_ids and ratings by the user_id column

        For example, given the input of:
        [(1, 101, 5.0), (1, 102, 3.0), (1, 103, 2.5), ...]

        The value of the accumulator would be:
        {1: [(101, 5.0), (102, 3.0), (103, 2.5)], ...}
        As we can see, these are the movies and ratings for user with id 1

        The accumulator is where the movie ratings will be stored for a given
        user. The accumulator should be a dictionary!

        The group_by_user_reducer function is going to be used in a reduction
        operation
    '''
    # unpack the tuple
    user_id, movie_id, rating = user_movie_rating

    # get the current movie ratings for the user_id, or an empty list if it
    # does not exist yet
    movie_ratings_for_user = accumulator.get(user_id, [])

    # append a 2-tuple of (movie_id, rating) to the movie ratings for this user
    movie_ratings_for_user.append((movie_id, rating))

    # set the user_id values so they use the move_ratings_for_user
    accumulator[user_id] = movie_ratings_for_user

    # return the accumulator (because this is a function which will be used in
    # a reducer
    return accumulator


def generate_user_occurance_matrix(movies_seen_by_user, size):
    ''' movies_seen_by_user is a list of movie ids which this user has rated.
        size is the size of the occurance matrix.

        user_occurance_matrix returns a sparse matrix which corresponds with
        the films that have co-occured for this user.

        The user_occurance_matrix has a value of 1 for movies that appear
        together. For instance, if the user has rated a movie with id A, and a
        movie with id B, then the user occurance matrix will have:
            user_occurance_matrix[A, B] == 1
            user_occurance_matrix[B, A] == 1

            - and because movies occur with themselves -

            user_occurance_matrix[A, A] == 1
            user_occurance_matrix[B, B] == 1

        The user_occurance_matrix allows us to see movies that co-occur
    '''

    # create the sparse matrix
    user_occurance_matrix = sparse_matrix((size, size), dtype=MAT_DTYPE)

    # we want the 2-size permutations of the movies seen by this user.
    # If movies_seen_by_user is [1, 2, 3], permutations will be
    # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    movie_id_permutations = permute(movies_seen_by_user, 2)
    for movie_id_1, movie_id_2 in movie_id_permutations:
        user_occurance_matrix[movie_id_1, movie_id_2] = 1

    # we also want to set it so that movies occur with themselves
    for movie_id in movies_seen_by_user:
        user_occurance_matrix[movie_id, movie_id] = 1

    # return the generated user_occurance_matrix
    return user_occurance_matrix


def generate_user_ratings_matrix(movie_ratings_for_user, size):
    ''' movie_ratings_for_user should be a list of 2-tuples with a format of:
            (movie_id, rating)
        the co-occurance matrix will be multiplied with this matrix to get the
        the movie recommendations for this user

        The returned matrix is an Nx1 matrix
    '''

    # generate an Nx1 matrix
    user_ratings_matrix = sparse_matrix((size, 1), dtype=MAT_DTYPE)

    # set the value of the movie_id to the rating
    for movie_id, rating in movie_ratings_for_user:
        user_ratings_matrix[movie_id, 0] = rating

    # return the matrix
    return user_ratings_matrix


# Filter out all empty lines from stdin and put them into input_lines
input_lines = filter(len, sys.stdin)

# catch all input errors, print to stderr and exit
try:
    # for every input line, parse the line and put the values into ratings
    # because python3 is < python2 I have to turn this into a list apparently
    ratings = list(map(parse_input_line, input_lines))
except InputError as e:
    # there was an InputError so record the error and exit
    print('Error with input line:', e.message, file=sys.stderr)
    sys.exit(1)

# gets the max movie_id. This is needed to compute the sparse matrix size
max_movie_id = max(ratings, key=lambda r: r[1])[1]
mat_size = max_movie_id + 1

# start with an empty co-occurance matrix
co_occurance_matrix = sparse_matrix((mat_size, mat_size), dtype=MAT_DTYPE)

''' perform a reduction using group_by_user_reducer to output a dictionary
    where the key of the dictionay is the user_id and the value is a list of
    movies and ratings for this user.

    ratings_grouped_by_user will look something like:
        {
            1: [(101, 3.0), (102, 4.0)],
            2: [(101, 4.0), (103, 2.0)],
            3: ...
        }
    In this example, user with id 1 has movie with id 101 a 3.0 rating, and has
    rated movie 102 a 4.0 rating.

    We need this data structure to know which movies have co-occured. In this
    example, movies with ids 101 and 102 have co-occured for user 1.

    We also need this data structure to recommend movies to this user.
'''
ratings_grouped_by_user = reduce(
    group_by_user_reducer,
    ratings,
    {}
)

''' First we need to generate the co-occurance matrix.
    To do this, first we get a sparse matrix which corresponds to the
    co-occurances of movies for a single user. Then we add this to the total
    co-occurance matrix. When we have performed this operation for all users we
    have the total co-occurance matrix which is used in the next step.
'''

for user_id, user_ratings in ratings_grouped_by_user.items():
    ''' From the above defined ratings_grouped_by_user, user_ratings will have
        a structure which looks like:
        [(101, 3.0), (102, 4.0)]

        To generate the co-occurance matrix, we only want to have the list of
        movie ids, we do not need the rating, so this map is performed, which
        will output a value which looks like this if we use the above example:
        [101, 102]
    '''
    user_movies = map(lambda ur: ur[0], user_ratings)

    ''' Now we generate the co-occurance matrix for this individual user.
        If we continue using the example above and create a co-occurance matrix
        for the list of movie ids [101, 102], then this co-occurance matrix
        will be a matrix where:

        matrix[101, 102] == 1
        matrix[102, 101] == 1
        matrix[101, 101] == 1
        matrix[102, 102] == 1
    '''
    user_occurance_matrix = generate_user_occurance_matrix(
        user_movies, mat_size
    )

    ''' Then we add this to the co_occurance_matrix to accumulate the results
        so we can get a total co-occurance matrix for all users
    '''
    co_occurance_matrix += user_occurance_matrix

''' Now that we have the co-occurance matrix we are ready to output
    recommendations for each user.
'''
for user_id, user_ratings in ratings_grouped_by_user.items():
    ''' user_ratings_matrix is a sparse Nx1 matrix, where N is the maximum
        movie id plus 1. So if the maximum movie id is 107, then this will be a
        108x1 sparse matrix. The value is the rating.

        If this user user_ratings are [(101, 5.0), (102, 3.0), ...], then this
        matrix will look like:

        [
            [0.0], # row with index 0
            [0.0], # row with index 1
            ...
            [5.0], # row with index 101
            [3.0], # row with index 102
            ...
        ]
    '''
    user_ratings_matrix = generate_user_ratings_matrix(user_ratings, mat_size)

    ''' To get the recommendations we must now multiply the co_occurance_matrix
        with the user_ratings_matrix. The result will be an Nx1 matrix where a
        high value corresponds to a high recommendation.
    '''
    recommendations_matrix = co_occurance_matrix * user_ratings_matrix

    ''' recommendation_items will have a structure which looks like:
        [((row, col), rating), (row, col), rating), ...]

        For example, this is what the recommendation_items looks like for user
        with id of 1 in the example for this assignment:

        [
            ((105, 0), 15.5),
            ((103, 0), 39.0),
            ((104, 0), 33.5),
            ((102, 0), 31.5),
            ((107, 0), 5.0),
            ((101, 0), 44.0),
            ((106, 0), 18.0)
        ]
    '''
    recommendation_items = recommendations_matrix.todok().items()

    ''' But, we only want the movie id and the recommendation value, so we
        perform this map to output a list of tuples correpsonding to the movie
        id and the recommendation value. Using the example above, the output
        is:

        [
            (105, 15.5),
            (103, 39.0),
            (104, 33.5),
            (102, 31.5),
            (107, 5.0),
            (101, 44.0),
            (106, 18.0)
        ]
    '''
    movie_recommendations = map(
        lambda i: (i[0][0], i[1]),
        recommendation_items
    )

    ''' movies_already_seen_list is a list of the movies that this user has
        already rated. Given a user_ratings of:
        [
            (101, 5.0),
            (102, 3.0),
            (103, 2.5),
        ]

        movies_already_seen_list would be:

        [101, 102, 103]
    '''
    movies_already_seen_list = map(lambda ur: ur[0], user_ratings)

    ''' I convert movies_already_seen_list to a set here because sets have O(1)
        for checking membership, whereas a list is O(N)
    '''
    movies_already_seen = set(movies_already_seen_list)

    ''' Now we filter out all recommendations if they are in the
        movies_already_seen set.
    '''
    unseen_movie_recommendations_iterator = filter(
        lambda mr: mr[0] not in movies_already_seen,
        movie_recommendations
    )

    # python3 returns an iterator so I must convert this to a list
    unseen_movie_recommendations = list(unseen_movie_recommendations_iterator)

    ''' We sort the unseen_movie_recommendations in descending order, making
        sure to sort by the recommendation value (lambda r: r[1]).
    '''
    unseen_movie_recommendations.sort(key=lambda r: r[1], reverse=True)

    ''' Just to make sure things don't break if there are no recommendations
        for this user.
    '''
    if len(unseen_movie_recommendations):
        recommendation = str(unseen_movie_recommendations[0][0])
    else:
        recommendation = '[NO RECOMMENDATION]'

    ''' Finally we print out the recommendation for this user, and add some
        text showing all of the recommendations for movies this user has not
        seen.
    '''
    print (
        'user %d: %s => recommend %s' % (
            user_id,
            str(unseen_movie_recommendations),
            recommendation
        )
    )
