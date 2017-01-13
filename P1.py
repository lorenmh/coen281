#!/opt/python-3.4/linux/bin/python3
#import numpy as np
import re, sys

SPLIT_RE_STR = r',\s?'
SPLIT_RE = re.compile(SPLIT_RE_STR)

class InputError(Exception):
    def __init__(self, message):
        self.message = message

def parse_input(input_line):
    input_list = SPLIT_RE.split(input_line)

    if len(input_list) != 3:
        raise InputError('Invalid input %s' % input_line)

    user_id_str, movie_id_str, rating_str = input_list

    user_id = int(user_id_str)
    movie_id = int(movie_id_str)
    rating = float(rating_str)

    return user_id, movie_id, rating

for line in sys.stdin:
    user_id, movie_id, rating = parse_input(line)
    print('USER_ID:%d\tMOVIE_ID:%d\tRATING:%f' % (user_id, movie_id, rating))
