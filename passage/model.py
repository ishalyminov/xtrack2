import cPickle
import sys
import logging


def flatten(l):
    return [item for sublist in l for item in sublist]


class NeuralModel(object):
    def __init__(self):
        self.params = []
        self.init_args = {}

    def save_params(self, f_name):
        obj = {}
        model_params = {}
        for param in self.params:
            model_params[param.name] = param.get_value()

        obj['model_params'] = model_params
        obj['init_args'] = self.init_args

        with open(f_name, 'w') as f_out:
            cPickle.dump(obj, f_out, -1)

    @classmethod
    def load(cls, f_name, **kwargs):
        with open(f_name) as f_in:
            obj = cPickle.load(f_in)

        init_args = obj['init_args']
        for arg, arg_val in kwargs.iteritems():
            init_args[arg] = arg_val

        m = cls(**init_args)
        m.update_params(obj['model_params'])

        return m

    def update_params(self, model_params):
        for param in sorted(self.params, key=lambda x: x.name):
            param_val = model_params.get(param.name)
            if param_val != None:
                logging.info('Loading param: %s' % param.name)
                assert param_val.shape == param.get_value().shape
                param.set_value(param_val)
            else:
                logging.info('Skipping param: %s' % param.name)

    def load_params(self, f_name):
        with open(f_name) as f_in:
            obj = cPickle.load(f_in)

        model_params = obj['model_params']
        self.update_params(model_params)


    def save(self, f_name):
        val = sys.getrecursionlimit()
        sys.setrecursionlimit(100000)
        with open(f_name, 'w') as f_out:
            cPickle.dump(self, f_out, -1)

        sys.setrecursionlimit(val)
