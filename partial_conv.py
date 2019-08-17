import tensorflow as tf
#import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

#weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
#weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


##################################################################################
# Layer
##################################################################################
def partial_conv(x, weights, strides, padding='SAME',scope='conv_0'):
    with tf.variable_scope(scope):
        if padding.lower() == 'SAME'.lower() :
            with tf.variable_scope('mask'):
                _, h, w, _ = x.get_shape().as_list()

                slide_window = int(weights.shape[0]) * int(weights.shape[1])
                mask = tf.ones(shape=[1, h, w, int(weights.shape[2])])

                update_mask = tf.nn.conv2d(mask, weights, strides=strides, padding=padding)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('x'):
                x = tf.nn.conv2d(x, weights, strides=strides, padding=padding)
                x = x * mask_ratio

        else :
            x = tf.nn.conv2d(x, weights, strides=strides, padding=padding)

        return x
