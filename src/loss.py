#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K

def nn_loss_all(penalty_coeff):
    """ Wrapper loss function

    Parameters
    ----------
        penalty_coeff: float
            Penalty coefficient for loss function

    Returns
    -------
        loss : func
            Loss function

    """

    penalty_coeff = K.constant(penalty_coeff)

    def loss(info, y_pred):
        """ Loss function

        Parameters
        ----------
            info: tensor
                Tensor containing knowledge on the train set
                and target value
            y_pred: tensor
                Tensor containing the actual prediction of
                the network

        Returns
        -------
            loss value : tensor
                Tensor representing the loss value


        """
        y_true = K.reshape(info[:, -1], (-1, 1))
        flags = K.reshape(info[:, -2], (-1, 1))
        kwb_matrix = info[:, 0:-2]
        # creating the constraint on the error
        rules = K.dot(K.transpose(kwb_matrix), y_pred)
        # remove random elements prediction
        where = tf.not_equal(flags, K.constant(0))
        y_pred = tf.boolean_mask(y_pred, where)
        y_true = tf.boolean_mask(y_true, where)
        return K.mean(K.square(y_pred - y_true)) + penalty_coeff * K.mean(rules)

    return loss


def nn_loss_violated(penalty_coeff):
    """ Wrapper loss function

    Parameters
    ----------
        penalty_coeff: float
            Penalty coefficient for loss function

    Returns
    -------
        loss : func
            Loss function

    """

    penalty_coeff = K.constant(penalty_coeff)

    def loss(info, y_pred):
        y_true = K.reshape(info[:, -1], (-1, 1))
        flags = K.reshape(info[:, -2], (-1, 1))
        kwb_matrix = info[:, 0:-2]
        # creating the constraint on the error
        rules = K.dot(K.transpose(kwb_matrix), y_pred)
        violated_rules = K.maximum(rules, K.constant(0))
        where = tf.not_equal(flags, K.constant(0))
        y_pred = tf.boolean_mask(y_pred, where)
        y_true = tf.boolean_mask(y_true, where)
        return K.mean(K.square(y_pred - y_true)) + penalty_coeff * K.mean(violated_rules)

    return loss
