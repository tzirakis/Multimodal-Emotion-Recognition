from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def concordance_cc(predictions, labels):
    """Complete me...

    Args:
    Returns:
    """
    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))

    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))


def concordance_cc2(r1, r2):
    """Complete me...

    Args:
    Returns:
    """
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()

    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)