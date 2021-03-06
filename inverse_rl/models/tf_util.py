import tensorflow as tf
import numpy as np

REG_VARS = 'reg_vars'
INDIV_VARS = 'indiv_vars'

def linear(X, dout, name, bias=True, individual_vars=False):
    with tf.variable_scope(name):
        def _get_var_and_register(var_name):
            var = tf.get_variable(var_name, shape=())
            tf.add_to_collection(INDIV_VARS, var)
            return var

        dX = int(X.get_shape()[-1])
        if individual_vars:
            W = tf.stack([[_get_var_and_register('W-%i-%i' % (i, j)) for j in range(dout)] for i in range(dX)])
        else:
            W = tf.get_variable('W', shape=(dX, dout))
        tf.add_to_collection(REG_VARS, W)
        if bias:
            b = tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32)))
        else:
            b = 0
    return tf.matmul(X, W)+b

def discounted_reduce_sum(X, discount, axis=-1):
    if discount != 1.0:
        disc = tf.cumprod(discount*tf.ones_like(X), axis=axis)
    else:
        disc = 1.0
    return tf.reduce_sum(X*disc, axis=axis)

def assert_shape(tens, shape):
    assert tens.get_shape().is_compatible_with(shape)

def relu_layer(X, dout, name, individual_vars=False):
    return tf.nn.relu(linear(X, dout, name, individual_vars=individual_vars))

def softplus_layer(X, dout, name, individual_vars=False):
    return tf.nn.softplus(linear(X, dout, name, individual_vars=individual_vars))

def tanh_layer(X, dout, name, individual_vars=False):
    return tf.nn.tanh(linear(X, dout, name, individual_vars))

def get_session_config():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    #session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return session_config


def load_prior_params(pkl_fname):
    import joblib
    with tf.Session(config=get_session_config()):
        params = joblib.load(pkl_fname)
    tf.reset_default_graph()
    #joblib.dump(params, file_name, compress=3)
    params = params['irl_params']
    #print(params)
    assert params is not None
    return params
