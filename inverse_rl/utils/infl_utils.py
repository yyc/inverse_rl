import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product as hvp
import itertools

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


def hessian(ys, xs, gradients_kwargs=None):
    """
    :param ys: a `Tensor` or a list of `Tensors`
    :param xs: a List of `Tensor`s
    :param gradients_kwargs: any kwargs needed to be passed to both gradient calls
    :return: n x n Hessian Matrix consisting of the derivative of `sum(ys)` wrt to each `x` in `xs`
    """

    if gradients_kwargs is None:
        gradients_kwargs = {}
    first_order_gradients = tf.stack(tf.gradients(ys, xs, stop_gradients=xs, **gradients_kwargs))

    n = len(xs)

    if tf.contrib.framework.is_tensor(ys):
        sample_size = tf.shape(ys)[0]
    else:
        sample_size = len(ys)

    sample_size = tf.cast(sample_size, tf.float32)
    # xs = tf.stack(xs)

    def hessian_compute_row(j, tensor_array):
        gradient = first_order_gradients[j]
        # Gradient is dy/dx1, dy/dx2, ...

        offset = j * n

        # TODO: Only compute half of the matrix, since it's symmetrical
        #second_order = tf.stack(tf.gradients(gradient, tf.slice(xs, 0, j+1)))
        # Column Loop
        # l_vars = [tf.constant(0, tf.int32), tensor_array]
        # _, tensor_array = tf.while_loop(
        #     lambda i, _: i < j + 1,
        #     lambda i, t_array: (i + 1, t_array
        #                                     .write(offset + i, second_order[i])
        #                                     .write(i * n + j, second_order[i])),
        #     l_vars
        # )

        second_order = tf.gradients(gradient, xs, stop_gradients = xs, **gradients_kwargs)
        for i in range(n):
            tensor_array = tensor_array.write(offset + i, tf.math.divide(second_order[i], sample_size))
        return j+1, tensor_array

    # Row Iterator
    # Declare an iterator and tensor array loop variables for the gradients.
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(xs[0].dtype, n * n)
    ]
    _, stacked_hessian = tf.while_loop(
        lambda j, _: j < n,
        hessian_compute_row,
        loop_vars
    )

    return tf.reshape(stacked_hessian.stack(), [n, n])


def tests():
    # Test Hessian
    in_1 = tf.constant(1., tf.float32)
    in_2 = tf.constant(1., tf.float32)
    in_3 = tf.constant(.1, tf.float32)
    a = tf.Variable(tf.ones(shape=[1]), dtype=tf.float32, name="a")
    b = tf.Variable(tf.ones(shape=[1]), dtype=tf.float32, name="b")
    c = tf.Variable(tf.ones(shape=[1]), dtype=tf.float32, name="c")
    d = in_1 * a * b
    e = in_2 * b * c
    f = in_3 * a * c
    L = d * e

    hessian_op = hessian(L, [a, b, c])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    hessian_val = sess.run(hessian_op)
    print(hessian_val)


if __name__ == "__main__":
    tests()