#!/opt/python-3.4/linux/bin/python3

'''****************************************************************************
Lauren Howard - 1/16/2017
COEN 281 - Assignment P1

This program outputs recommendations using the co-occurance technique specified
in the P1 assignment guidelines document.

For all matrices, a scipy sparse matrix is used to minimize memory overhead.

First, all input is parsed from stdin. The program expects values to be comma
delimited. If there are any errors with the input then the error is printed to
stderr and the program exits with code 1.

Then all of the ratings are grouped by user.

Using these ratings a co-occurance matrix is created for each user.

The user co-occurance matrices are accumulated into a single matrix which is
the co-occurance matrix used to output recommendation values.

Then a single column matrix is created for each user. The values in this matrix
are the movie ratings for the user.

To get the recommendations for a user, we multiply the total co-occurance
matrix with that user's ratings matrix.

Then, we filter out the recommendations for movies the user has already seen
and sort the recommendations in descending order.

The highest valued recommendation is the recommendation for this user.

Given the example input from the assignment, the output should look like:
user 1: [(104, 33.5), (106, 18.0), (105, 15.5), (107, 5.0)] => recommend 104
user 2: [(106, 20.5), (105, 15.5), (107, 4.0)] => recommend 106
user 3: [(103, 26.5), (102, 20.0), (106, 17.5)] => recommend 103
user 4: [(102, 37.0), (105, 26.0), (107, 9.5)] => recommend 102
user 5: [(107, 11.5)] => recommend 107
****************************************************************************'''

import re
import sys
import warnings

import numpy as np
import scipy as sp

from functools import reduce
from itertools import permutations as permute

from scipy.sparse import csc_matrix as sparse_matrix


''' Input regular expression.
    I've annotated it and have split it into multiple lines to try to make it
    more readable. See this link if you would like to see the regular
    expression in action: https://regex101.com/r/n2sEnm/1
'''
INPUT_RE_STR = (
    # match start of string
    r'^' +
    # match a comment which might be preceded by spaces, OR
    r'(?:\s*#.*)|' +
    # match a non-capturing group
    r'(?:'
    # there can be 0 or more spaces followed by a capturing group named user_id
    # which matches 1 or more digits, which is followed by 0 or more spaces and
    # a comma
    r'\s*(?P<user_id>\d+)\s*,' +
    # there can be 0 or more spaces followed by a capturing group named
    # movie_id which matches 1 or more digits, which is followed by 0 or more
    # spaces and a comma
    r'\s*(?P<movie_id>\d+)\s*,' +
    # there can be 0 or more spaces followed by a capturing group named rating
    # which matches 1 or more digits which might be followed by a decimal point
    # and 1 or more digits (if its a float), which is followed by 0 or more
    # spaces
    r'\s*(?P<rating>\d+(?:\.\d+)?)\s*' +
    # which might be followed by a comment
    r'(?:#.*)?' +
    # close out the non-capturing group
    r')' +
    # match end of string
    r'$'
)
INPUT_RE = re.compile(INPUT_RE_STR)


# any float type is acceptable here, I am using 64 bit because I am running
# this on a 64 bit machine.
MAT_DTYPE = np.float64


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
    # a reducer)
    return accumulator


def generate_user_occurance_matrix(movies_seen_by_user, size):
    ''' movies_seen_by_user is a list of movie ids which this user has rated.
        size is the size of the occurance matrix.

        generate_user_occurance_matrix returns a sparse matrix which
        corresponds with the films that have co-occured for this user.

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

# This is where the magic begins

# catch all input errors, print to stderr and exit if an InputError is caught
try:
    # for every input line, parse the line and put the values into ratings
    ratings = list(parsed_input_generator(sys.stdin))
except InputError as e:
    # there was an InputError so record the error and exit
    print('Error with input:', e.message, file=sys.stderr)
    sys.exit(1)

# gets the max movie_id. This is needed to compute the sparse matrix size
# movie_id is in the second column which is why you see the two `[1]` accesses
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
    In this example, user with id 1 has rated movie with id 101 a 3.0 rating,
    and has rated movie 102 a 4.0 rating.

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
        movie ids, we do not need the rating, so this map is performed which
        will output a value which looks like this if we use the above example:
        [101, 102]
    '''
    movies_rated_by_user = map(lambda ur: ur[0], user_ratings)

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
        movies_rated_by_user, mat_size
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

        If this user's user_ratings are [(101, 5.0), (102, 3.0), ...], then
        this matrix will look like:

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

        matrix.todok() converts a sparse matrix to a dictionary
        dictionary.items() converts a dictionary to a list of key, value pairs
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
        sure to sort by the recommendation value (lambda mr: mr[1]).
    '''
    unseen_movie_recommendations.sort(key=lambda mr: mr[1], reverse=True)

    ''' Just to make sure things don't break if there are no recommendations
        for this user.
    '''
    if len(unseen_movie_recommendations):
        ''' the first item of unseen_movie_recommendations is the
            (movid_id, rating) tuple of the highest rated movie. We only want
            the movie_id which is why you see the list access of [0][0] here
        '''
        recommendation = str(unseen_movie_recommendations[0][0])
    else:
        # there are no recommendations for this user
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
