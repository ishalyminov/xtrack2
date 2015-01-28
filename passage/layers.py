import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import shared0s, flatten
import activations
import inits
import costs

import numpy as np

def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot

srng = RandomStreams()

class Layer(object):
    name = "unnamed_layer"
    #def connect(self):
    #    pass

    def output(self, dropout_active=False):
        raise NotImplementedError()

    def _name_param(self, param_name):
        return "%s__%s" % (self.name, param_name, )



class MatrixInput(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self.size = matrix.shape[-1]

    def output(self, dropout_active=False):
        return T.as_tensor(self.matrix)

    def get_params(self):
        return set()


class IdentityInput(object):
    def __init__(self, val, size):
        self.val = val
        self.size = size

    def output(self, dropout_active=False):
        return self.val

    def get_params(self):
        return set()



class Embedding(Layer):

    def __init__(self, name=None, size=128, n_features=256, init='normal'):
        if name:
            self.name = name
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.input = T.imatrix()
        self.wv = self.init((self.n_features, self.size),
                            layer_width=self.size,
                            scale=1.0,
                            name=self._name_param("emb"))
        self.params = {self.wv}

    def output(self, dropout_active=False):
        return self.wv[self.input]

    def get_params(self):
        return self.params



class LstmRecurrent(Layer):

    def __init__(self, name=None, size=256, init='normal', truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False):
        if name:
            self.name = name
        self.init = getattr(inits, init)
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        # Input connections.
        self.w = self.init((self.n_in, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("W"))

        self.b = self.init((self.size * 4, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("b"))

        # Initialize forget gates to large values.
        b = self.b.get_value()
        b[:self.size] = np.random.uniform(low=40.0, high=50.0, size=self.size)
        self.b.set_value(b)

        # Recurrent connections.
        self.u = self.init((self.size, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("U"))

        # Peep-hole connections.
        self.p = self.init((self.size, self.size * 3),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("P"))

        self.params = [self.w, self.u, self.p, self.b]

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_t, h_tm1, c_tm1, u, p):
        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u

        g_f = self._slice(gates_fiom, 0)
        g_i = self._slice(gates_fiom, 1)
        g_o = self._slice(gates_fiom, 2)
        g_m = self._slice(gates_fiom, 3)

        c_tm1_dot_p = T.dot(c_tm1, p)

        g_f += self._slice(c_tm1_dot_p, 0)
        g_i += self._slice(c_tm1_dot_p, 1)
        g_o += self._slice(c_tm1_dot_p, 2)

        g_f = T.nnet.sigmoid(g_f)
        g_i = T.nnet.sigmoid(g_i)
        g_o = T.nnet.sigmoid(g_o)
        g_m = T.tanh(g_m)

        c_t = g_f * c_tm1 + g_i * g_m
        h_t = g_o * T.tanh(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop

        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        [out, cells], _ = theano.scan(self.step,
            sequences=[x_dot_w],
            outputs_info=[T.alloc(0., X.shape[1], self.size), T.alloc(0., X.shape[1], self.size)], 
            non_sequences=[self.u, self.p],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            if self.out_cells:
                return cells
            else:
                return out
        else:
            if self.out_cells:
                return cells[-1]
            else:
                return out[-1]

    def get_params(self):
        return self.l_in.get_params().union(self.params)


class Dense(Layer):
    def __init__(self, name=None, size=256, activation='rectify', init='normal',
                 p_drop=0.):
        if name:
            self.name = name
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init(
            (self.n_in, self.size),
            layer_width=self.size,
            name=self._name_param("w")
        )
        self.b = self.init(
            (self.size, ),
            layer_width=self.size,
            name=self._name_param("b")
        )
        self.params = [self.w, self.b]

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.
        else:
            dropout_corr = 1.0 - self.p_drop

        is_tensor3_softmax = X.ndim > 2 and self.activation_str == 'softmax'

        shape = X.shape
        if is_tensor3_softmax: #reshape for tensor3 softmax
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w * dropout_corr) + self.b)

        if is_tensor3_softmax: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

    def get_params(self):
        return set(self.params).union(set(self.l_in.get_params()))


class MLP(Layer):
    def __init__(self, sizes, activations, name=None, p_drop=0.):
        layers = []
        for layer_id, (size, activation) in enumerate(zip(sizes, activations)):
            layer = Dense(size=size, activation=activation, name="%s_%d" % (
                name, layer_id, ), p_drop=p_drop)
            layers.append(layer)

        self.stack = Stack(layers, name=name)

    def connect(self, l_in):
        self.stack.connect(l_in)

    def output(self, dropout_active=False):
        return self.stack.output(dropout_active=dropout_active)

    def get_params(self):
        return set(self.stack.get_params())


class Stack(Layer):
    def __init__(self, layers, name=None):
        if name:
            self.name = name
        self.layers = layers
        self.size = layers[-1].size


    def connect(self, l_in):
        self.layers[0].connect(l_in)
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])

    def output(self, dropout_active=False):
        return self.layers[-1].output(dropout_active=dropout_active)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))


class CherryPick(Layer):
    def connect(self, data, indices, indices2):
        self.data_layer = data
        self.indices = indices
        self.indices2 = indices2
        self.size = data.size

    def output(self, dropout_active=False):
        out = self.data_layer.output(dropout_active=dropout_active)
        return out[self.indices, self.indices2]

    def get_params(self):
        return set(self.data_layer.get_params())



class CrossEntropyObjective(Layer):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)

        return costs.CategoricalCrossEntropy(self.y_true,
                                             y_hat_out)

    def get_params(self):
        return set(self.y_hat_layer.get_params())


class SumOut(Layer):
    def connect(self, *inputs, **kwargs):
        self.inputs = inputs
        self.scale = kwargs.get('scale', 1.0)

    def output(self, dropout_active=False):
        res = 0
        for l_in in self.inputs:
            res += l_in.output(dropout_active)

        return res * self.scale

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.inputs]))