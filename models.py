from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim

def recurrent_model(net, hidden_units=256, number_of_outputs=2):

#    with tf.variable_scope("recurrent"):
      batch_size, seq_length, num_features = net.get_shape().as_list()

      lstm1 = tf.contrib.rnn.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      lstm2 = tf.contrib.rnn.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

      # We have to specify the dimensionality of the Tensor so we can allocate
      # weights for the fully connected layers.
      outputs, states = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

      net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

      prediction = slim.layers.linear(net, number_of_outputs)

      return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))


def video_model(video_frames=None, audio_frames=None):
    """Complete me...

    Args:
    Returns:
    """

    with tf.variable_scope("video_model"):
        batch_size, seq_length, height, width, channels = video_frames.get_shape().as_list()

        video_input = tf.reshape(video_frames, (batch_size * seq_length, height, width, channels))
        video_input = tf.cast(video_input, tf.float32)

        features, end_points = resnet_v1.resnet_v1_50(video_input, None)
        features = tf.reshape(features, (batch_size, seq_length, int(features.get_shape()[3])))

    return features

def audio_model(video_frames=None, audio_frames=None, conv_filters=40):
    """Complete me...

    Args:
    Returns:
    """

    with tf.variable_scope("audio_model"):
      print(audio_frames.get_shape().as_list())

      batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
      audio_input = tf.reshape(audio_frames, [batch_size * seq_length, 1, num_features, 1])

      with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
        net = slim.dropout(audio_input)
        net = slim.layers.conv2d(net, conv_filters, (1, 20))

        # Subsampling of the signal to 8KhZ.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name='pool1')

        # Original model had 400 output filters for the second conv layer
        # but this trains much faster and achieves comparable accuracy.
        net = slim.layers.conv2d(net, conv_filters, (1, 40))

        net = tf.reshape(net, (batch_size * seq_length, num_features // 2, conv_filters, 1))

        # Pooling over the feature maps.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 10, 1],
            strides=[1, 1, 10, 1],
            padding='SAME',
            name='pool2')

      net = tf.reshape(net, (batch_size, seq_length, num_features //2 * 4 ))

    return net


def combined_model(video_frames, audio_frames):
    """Complete me...

    Args:
    Returns:
    """
    audio_features = audio_model([], audio_frames)
    visual_features = video_model(video_frames,[])

    return tf.concat((audio_features, visual_features), 2, name='concat')


def get_model(name):
    """Complete me...

    Args:
    Returns:
    """
    name_to_fun = {'audio': audio_model, 'video': video_model, 'both': combined_model}

    if name in name_to_fun:
        model = name_to_fun[name]
    else:
        raise ValueError('Requested name [{}] not a valid model'.format(name))

    def wrapper(*args, **kwargs):
        return recurrent_model(model(*args), **kwargs)

    return wrapper
