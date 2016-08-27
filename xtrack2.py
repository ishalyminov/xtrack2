import codecs
import json
import os
import logging
import random
import theano
import theano.gradient
import time
import argparse

from data import Data
from utils import pdb_on_error
from model import Model
from model_simple_conv import SimpleConvModelKeras
from model_baseline import BaselineModelKeras

theano.config.floatX = 'float32'
# theano.config.allow_gc=False
# theano.scan.allow_gc=False
# theano.config.profile=True
# theano.config.mode = 'FAST_COMPILE'
# theano.config.linker = 'py'
# theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_compile'


def init_output_dir(out_dir):
    cntr = 1
    orig_out_dir = out_dir
    while os.path.exists(out_dir):
        out_dir = orig_out_dir + '_' + str(cntr)
        cntr += 1

    os.makedirs(out_dir)
    return out_dir


def init_env(output_dir):
    output_dir = init_output_dir(output_dir)

    theano.config.compute_test_value = 'off'
    theano.config.allow_gc=False
    theano.scan.allow_gc=False

    # Setup logging.
    logger = logging.getLogger('XTrack')
    logger.setLevel(logging.DEBUG)

    logging_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(logging_format)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.root = logger

    return output_dir


def get_model(in_args, in_train_data):
    with codecs.getreader('utf-8')(open(in_args.model_config_file)) as input:
        model_config = json.load(input)
    n_input_tokens = len(in_train_data.vocab)
    slots = in_train_data.slots

    if in_args.model_type == 'lstm':
        return Model(slots, in_train_data.classes, model_config)
    if in_args.model_type == 'conv':
        return SimpleConvModelKeras(
            slots,
            in_train_data.classes,
            in_train_data.vocab,
            model_config,
            os.path.join(
                in_args.out + os.path.basename(in_args.experiment_path),
                'conv.h5'
            )
        )
    if in_args.model_type == 'baseline':
        return BaselineModelKeras(
            slots,
            in_train_data.classes,
            in_train_data.vocab,
            model_config,
            os.path.join(
                in_args.out + os.path.basename(in_args.experiment_path),
                'baseline.h5'
            )
        )
    raise RuntimeError('"{}" - unknown model type'.format(in_args.model_type))


def main(in_args):
    output_dir = \
        init_env(in_args.out + os.path.basename(in_args.experiment_path))
    logging.info('XTrack has been started.')
    logging.info('Output dir: %s' % output_dir)
    logging.info('Initializing random seed to 271.')
    random.seed(271)
    logging.info('Experiment path: %s' % in_args.experiment_path)

    train_path = os.path.join(in_args.experiment_path, 'train.json')
    xtd_t = Data.load(train_path)

    valid_path = os.path.join(in_args.experiment_path, 'dev.json')
    xtd_v = Data.load(valid_path)

    t = time.time()

    model = get_model(in_args, xtd_t)
    X, y = model.prepare_data_train(xtd_t.sequences, xtd_t.slots)
    X_valid, y_valid = model.prepare_data_test(xtd_v.sequences, xtd_v.slots)

    logging.info('Building model: %s' % in_args.model_type)
    model.train(X, y)
    logging.info('Training took: %.1f' % (time.time() - t))

    logging.info('Result model saved as "{}"'.format(model.save_path))
    model.save(model.save_path)
    # tracker_valid = XTrack2DSTCTracker(xtd_v, [model])


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument(
        '--model_config_file',
        required=True
    )
    result.add_argument('experiment_path')

    result.add_argument('--eid', default='xtrack_experiment')
    result.add_argument('--valid_after', default=None, type=int)

    # Experiment params.
    result.add_argument('--load_params', default=None)
    result.add_argument('--save_params', default=None)
    result.add_argument('--out', default='xtrack2_out')

    result.add_argument('--model_type', default='lstm', type=str)

    result.add_argument('--debug', action='store_true')
    result.add_argument('--track_log', default='track_log.txt', type=str)

    return result

if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    pdb_on_error()

    main(args)
