import numpy as np

def standardize_normal(arr):
    # every value is the z-score of the original value
    return [(x - np.mean(arr)) / np.std(arr) for x in arr]

def normalize(arr):
    return [(x - np.min(arr)) / float(np.max(arr) - np.min(arr)) for x in arr]