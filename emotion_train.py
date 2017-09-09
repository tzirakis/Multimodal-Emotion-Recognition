from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_provider
import losses
import models

from tensorflow.python.platform import tf_logging as logging


slim = tf.contrib.slim

# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97, 'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('batch_size', 2, 'The batch size to use.')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           'Directory where to write event logs '
                           'and checkpoint.')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
			   'If specified, restore this pretrained model '
                           'before beginning any training.')
tf.app.flags.DEFINE_integer('hidden_units', 256, 
			    'The number of hidden units in the recurrent model')
tf.app.flags.DEFINE_integer('seq_length', 2, 
			    'The number of consecutive examples to be used' 
			    'in the recurrent model')
tf.app.flags.DEFINE_string('model', 'both',
                           'Which model is going to be used: audio, video, or both ')
tf.app.flags.DEFINE_string('dataset_dir', 'path_to_tfrecords',
                           'The tfrecords directory.')

def train(data_folder):

    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        frames, audio, ground_truth, _ = data_provider.get_split(data_folder, True,
                                                                 'train', FLAGS.batch_size,
						                  seq_length=FLAGS.seq_length)

        # Define model graph.
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                is_training=True):
          with slim.arg_scope(slim.nets.resnet_utils.resnet_arg_scope(is_training=True)):
                  prediction = models.get_model(FLAGS.model)(frames, audio,
                                                            hidden_units=FLAGS.hidden_units)

        for i, name in enumerate(['arousal', 'valence']):
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,))

            loss = losses.concordance_cc(pred_single, gt_single)
            tf.summary.scalar('losses/{} loss'.format(name), loss)

            mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            tf.summary.scalar('losses/mse {} loss'.format(name), mse)

            slim.losses.add_loss(loss / 2.)

        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

        init_fn = None
        with tf.Session(graph=g) as sess:
            if FLAGS.pretrained_model_checkpoint_path:
		# Need to specify which variables to restore (use scope of models)
                variables_to_restore = slim.get_variables()
                init_fn = slim.assign_from_checkpoint_fn(
                        FLAGS.pretrained_model_checkpoint_path, variables_to_restore)

            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)


            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                FLAGS.train_dir,
                                init_fn=init_fn,
                                save_summaries_secs=60,
                                save_interval_secs=300)


if __name__ == '__main__':
    train(FLAGS.dataset_dir)
