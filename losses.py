from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim

def concordance_cc(prediction, ground_truth):
    """Defines concordance loss for training the model. 
    
    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """

    pred_mean, pred_var = tf.nn.moments(prediction, (0,))
    gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))

    mean_cent_prod = tf.reduce_mean((prediction - pred_mean) * (ground_truth - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))
