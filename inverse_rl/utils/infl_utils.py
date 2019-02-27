import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product as hvp


# Approximats H^-1 v wrt weights
def hessian_inverse_v(loss_samples, weights, v):

    def func(previous_h_inv, sample):
        return v + previous_h_inv - hvp(sample, weights, previous_h_inv)
    return tf.scan(func, loss_samples, initializer=v)

def hessian_compute_inverse_v(loss_samples, weights, v, feed_dict={}):
    samples = tf.split(loss_samples, loss_samples.shape[0], axis=0)
    h_inv = v
    counter = 0
    for s in samples:
        print('infl itr {}'.format(counter))
        counter = counter + 1
        new_hvp = tf.get_default_session().run(hvp(s, weights, h_inv), feed_dict=feed_dict)
        h_inv = np.add(np.add(v, h_inv), np.multiply(-1, new_hvp))
    return h_inv

def hessian_compute_inverse_vs(loss_samples, weights, vs, feed_dict={}):
    samples = tf.split(loss_samples, loss_samples.shape[0], axis=0)
    n = len(vs)
    log = [vs]
    h_invs = vs
    counter = 0
    for s in samples:
        print('infl itr {}'.format(counter))
        counter = counter + 1
        new_hvps_ops = [hvp(s, weights, h_inv) for h_inv in h_invs]
        new_hvps = tf.get_default_session().run(new_hvps_ops, feed_dict=feed_dict)
        h_invs = np.add(np.add(vs, h_invs), np.multiply(-1, new_hvps))
        log.append(h_invs)
    return log, h_invs