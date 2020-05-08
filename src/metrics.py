#!/usr/bin/env python # -*- coding: utf-8 -*-

import numpy as np

def mae(y_true, y_pred):
    """ Compute Mean Absolute Error

    This function computes MAE on the non log
    error

    Parameters
    ----------
        y_true : list(float)
            True value for
            a given sample of data
        y_pred : list(float)
            Predicted value for
            a given sample of data

    Returns
    -------
        MAE : float
            Mean Absolute Error

    """
    y_pred = np.array([10 ** -y for y in y_pred])
    y_true = np.array([10 ** -y for y in y_true])
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    """ Compute Root Mean Squared Error

    This function computes MAE on the non log
    error

   Parameters
   ----------
       y_true : list(float)
           True value for
           a given sample of data
       y_pred : list(float)
           Predicted value for
           a given sample of data

   Returns
   -------
       RMSE : float
           Root Mean Squared Error

       """
    y_pred = np.array([-10 ** y for y in y_pred])
    y_true = np.array([-10 ** y for y in y_true])
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def is_dominant(x, y):
    """ Checks if the configuration x is dominant over y

    Parameters
    ----------
        x : list(float)
            configuration
        y : list(float)
            configuration

    Returns
    -------
        Dominance Truth Value : bool
            True if x is dominant over y, False otherwise

    """
    n = len(x) if isinstance(x, list) else x.shape[0]
    return all([x[i] > y[i] for i in range(n)])


def const_violation(accuracy, error):
    """ Check how many constraint are violated in a prediction

    Parameters
    ----------
        accuracy : list(list(float)
            matrix of configuration that were predicted
        error : list(float)
            error predicted

    Returns
    -------
        Percentage : float
            Percentage of violated constraint over all the
            possible couples of configurations
        Equal prediction : int
            Number of prediction repeated in the output vector
            (this is used in order to verify if the model is
            well trained)

    """
    accuracy = list(accuracy)
    error = list(error)
    n = len(accuracy)
    count = 0
    tot = 0
    for i in range(n-1):
        for j in range(i+1, n):
            tot += 1
            if is_dominant(accuracy[i], accuracy[j]):
                count += 1 if error[i] < error[j] else 0

    u, c = np.unique(error, return_counts=True)
    dup = np.sum(c[c > 1])

    return (count/tot * 100) , dup, tot


def error_average_distance(y):
    """ Average distance between istances

    Parameters
    ----------
        y : list(float)
            Error value for a set of configurations

    Returns
    -------
        Distance : float
            Average distace among errors
    """
    couple_distance = []
    for i in range(len(y)-1):
        for j in range(i, len(y)):
            couple_distance.append(np.sqrt(np.square(y[i] - y[j])))

    return np.average(np.array(couple_distance))
