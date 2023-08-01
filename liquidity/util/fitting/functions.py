import numpy as np


def exponential_function(x, beta, lambda_):
    return beta * np.exp(-lambda_ * x)


def logarithmic_function(x, beta, lambda_):
    return beta + lambda_ * np.log(x)


def periodic_function(x, a0, A, omega):
    return a0 + A * np.sin(omega * x)


def linear_function(x, beta, lambda_):
    return beta + lambda_ * x
