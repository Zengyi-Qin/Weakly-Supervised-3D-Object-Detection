import tensorflow as tf

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  =  tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
                    tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

def cross_entropy(prob, label):
    cnt = label * tf.log(prob + 1e-6) + (1-label) * tf.log(1 - prob + 1e-6)
    return -cnt