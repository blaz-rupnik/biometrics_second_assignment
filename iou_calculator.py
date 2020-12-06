import numpy as np


# inspired by https://stackoverflow.com/questions/49338166/python-intersection-over-union
# array1 and array2: 0 -> ear not detected there, 1 -> ear detected there
def ioc_calculator(array1, array2):
    overlap = array1 * array2 # AND
    union = array1 + array2 # OR
    return overlap.sum()/float(union.sum())