from collections import defaultdict
import itertools
import json
import os
import sys
import logging
import numpy as np
import random
import theano
import theano.gradient
import time

theano.config.floatX = 'float32'
theano.config.allow_gc=False
theano.scan.allow_gc=False
#theano.config.profile=True
#theano.config.mode = 'FAST_COMPILE'
#theano.config.linker = 'py'
theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'None'

from passage.iterators import (padded, SortedPadded)
from passage.utils import iter_data
from data import Data, DataBuilder
from utils import (get_git_revision_hash, pdb_on_error, ConfusionMatrix, P,
                   inline_print, update_progress)
from mlpmodel import Model
from model_simple_conv import SimpleConvModel
from model_baseline import BaselineModel
from model_turn_based import TurnBasedModel
from model_turn_based_rnn import TurnBasedRNNModel
from dstc_tracker import XTrack2DSTCTracker
from templates import TemplateExtractor



def plot_loss(model, data):
    params = []

    directions_x = []
    directions_y = []
    for th_param in model.params:
        param = th_param.get_value()
        directions_x.append(np.array(np.random.rand(*param.shape), dtype='float32'))
        directions_y.append(np.array(np.random.rand(*param.shape), dtype='float32'))
        params.append(np.array(param))

    """
    res = []
    for x in np.linspace(-0.020, 0.-0.010, num=100):
        for th_param, param, dx, dy in zip(model.params, params, directions_x, directions_y):
            th_param.set_value(param + x * dx)

        loss = model._loss(*data)
        res.append((x, loss, ))
        print x, loss

    import matplotlib.pyplot as plt
    x, y = zip(*res)
    plt.plot(x, y)
    plt.show()"""

    res = []
    for x in np.linspace(-1.0, 0.0, num=100):
        for y in np.linspace(-1.0, 0.0, num=100):
            for th_param, param, dx, dy in zip(model.params, params, directions_x, directions_y):
                th_param.set_value(param + x * dx + y * dy)

            loss = model._loss(*data) / len(data[-1])
            print x, y, loss
            res.append((x, y, loss, ))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    x, y, temp = map(np.array, zip(*res))
    nrows, ncols = 10, 10
    grid = temp.reshape((nrows, ncols))

    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
               interpolation='nearest', cmap=cm.Blues)
    plt.show()





def compute_stats(slots, slot_selection, classes, prediction, y,
                  joint_slot_name):
    conf_mats = {}
    conf_mats[joint_slot_name] = ConfusionMatrix(2)
    for slot in slots:
        if slot in slot_selection:
            conf_mats[slot] = ConfusionMatrix(len(classes[slot]))


    joint_correct = np.array([True for _ in prediction[0]])
    joint_all = np.array([True for _ in prediction[0]])
    for i, (slot, pred) in enumerate(zip(slots, prediction)):
        if slot in slot_selection:
            slot_y = y[i]
            slot_y_hat = np.argmax(pred, axis=1)

            conf_mats[slot].batchAdd(slot_y, slot_y_hat)

            joint_correct &= (slot_y == slot_y_hat)

    conf_mats[joint_slot_name].batchAdd(joint_all, joint_correct)
    return conf_mats


def print_mb(slots, classes, vocab_rev, mb, prediction):
    x, x_score, x_switch, x_actor, y_seq_id, y_time, y_labels = mb

    labels = {}
    pred_id = {}
    for i, (seq_id, time), lbls in enumerate(zip(zip(y_seq_id, y_time),
                                              *y_labels)):
        labels[(seq_id, time)] = lbls
        pred_id[(seq_id, time)] = i

    example = []
    for dialog_id, dialog in enumerate(zip(*x)):
        print
        for t, w in enumerate(dialog):
            print vocab_rev[w],

            curr_ndx = (dialog_id, t)
            if curr_ndx in labels:
                curr_label = labels[curr_ndx]
                curr_pred = [prediction[i][pred_id[curr_ndx]]
                             for i, _ in enumerate(slots)]

                print
                print curr_label
                print curr_pred




def visualize_prediction(xtd, prediction):
    #x = data['x'].transpose(1, 0)
    pred_ptr = 0

    classes_rev = {}
    for slot in xtd.slots:
        classes_rev[slot] = {val: key
                             for key, val in xtd.classes[slot].iteritems()}

    for d_id, dialog in enumerate(xtd.sequences[:3]):
        print ">> Dialog %d" % d_id, "=" * 30

        labeling = {}
        for label in dialog['labels']:
            pred_label = {}
            for i, slot in enumerate(xtd.slots):
                pred = prediction[i][pred_ptr]
                pred_label[slot] = np.argmax(pred)
            pred_ptr += 1

            labeling[label['time']] = (label['slots'], pred_label)

        print " T:",
        last_score = None
        for i, word_id, score in zip(itertools.count(),
                                     dialog['data'],
                                     dialog['data_score']):
            if score != last_score:
                print "%4.2f" % score,
                last_score = score

            print xtd.vocab_rev[word_id], #"(%.2f)" % score,
            if i in labeling:
                print
                for slot in xtd.slots:
                    lbl, pred_lbl = labeling[i]
                    p = P()
                    p.print_out("    * ")
                    p.print_out(slot)
                    p.tab(20)
                    p.print_out(classes_rev[slot][lbl[slot]])
                    p.tab(40)
                    p.print_out(classes_rev[slot][pred_lbl[slot]])
                    print p.render()
                print " U:",
        print



def vlog(txt, *args, **kwargs):
    separator = kwargs.pop('separator', '\n')
    res = [txt]
    for k, v in args:
        res.append('\t%s(%s)' % (k, str(v)))
    for k, v in sorted(kwargs.iteritems()):
        res.append('\t%s(%s)' % (k, str(v)))
    logging.info(separator.join(res))


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

    logging.root = logger  #

    return output_dir


def prepare_minibatches(seqs, mb_size, model, slots):
    minibatches = []
    seqs_mb = iter_data(seqs, size=mb_size)
    for mb in seqs_mb:
        data = model.prepare_data_train(mb, slots, dense_labels=False)
        minibatches.append(data)

    return minibatches

def compute_prt(cmat, i):
    r = cmat[i,i] * 100.0
    total_i = cmat[i, :].sum()
    if total_i > 0:
        r /= total_i
    else:
        r = 100.0
    p = cmat[i,i] * 100.0
    total_ii = cmat[:, i].sum()
    if total_ii > 0:
        p /= total_ii
    else:
        p = 100.0

    return p, r, total_i


def visualize_mb(model, mb):
    prediction = model._predict(*mb)

    #print_mb(xtd_v, prediction_valid)



class TrainingStats(object):
    def __init__(self):
        self.data = defaultdict(list)

    def insert(self, **kwargs):
        for key, val in kwargs.iteritems():
            if type(val) is np.ndarray:
                val = float(val)
            self.data[key].append(val)

    def mean(self, arg):
        return np.array(self.data[arg]).mean()


def _get_example_list(minibatches, sorted_items, xtd_t):
    examples = []
    for ii, (i, loss) in enumerate(sorted_items):
        _, d = minibatches[i]

        x, x_score, x_switch, x_actor, y_seq_id, y_time, y_labels = d

        example = []
        for d in zip(*x):
            ln = ""
            for w in d:
                ln += xtd_t.vocab_rev[w]
                ln += " "
            example.append(ln)
        examples.append(example)
    return examples


def get_extreme_examples(mb_loss, minibatches, xtd_t):
    sorted_items = sorted(mb_loss.items(), key=lambda e: -e[1])
    worst_mb_ndxs = sorted_items[:5]
    worst_examples = _get_example_list(minibatches, worst_mb_ndxs, xtd_t)
    best_mb_ndxs = sorted_items[-5:]
    best_examples = _get_example_list(minibatches, best_mb_ndxs, xtd_t)

    return (worst_examples, worst_mb_ndxs), (best_examples, best_mb_ndxs)


def add_train_noise(mb_data, train_noise, vocab_size):
    res = list(mb_data)
    x = res[0] = np.array(res[0])

    d_lens = (x[:, :, 0] > 0).sum(axis=0)
    d_map = []
    d_pos_map = []

    curr_len = 0
    curr_d = 0
    for i in range(d_lens.sum()):
        d_map.append(curr_d)
        d_pos_map.append(curr_len)
        curr_len += 1

        if curr_len == d_lens[curr_d]:
            curr_d += 1
            curr_len = 0

    n_corrupt = (x.shape[0] * x.shape[1]) * train_noise
    curr_corrupt = 0
    while curr_corrupt < n_corrupt:
        t = random.randint(0, len(d_map) - 1)
        d_id = d_map[t]
        t_id = d_pos_map[t]
        x[t_id, d_id, 0] = random.randint(0, vocab_size - 1)

        curr_corrupt += 1

    #import ipdb; ipdb.set_trace()

    return res




def main(args_lst,
         eid, experiment_path, out, valid_after,
         load_params, save_params,
         debug, debug_data, debug_theano,
         n_cells, emb_size, x_include_score, no_train_emb, ftr_emb_size, token_features,
         split_cost, train_noise,
         n_epochs, lr, lr_anneal_factor, opt_type, momentum,
         n_early_stopping, early_stopping_group,
         mb_size, mb_mult_data,
         oclf_n_hidden, oclf_n_layers, oclf_activation,
         rnn_n_layers,
         lstm_type, lstm_update_thresh, lstm_peepholes, lstm_bidi,
         p_drop, init_emb_from, input_n_layers, input_n_hidden,
         input_activation,
         eval_on_full_train, x_include_token_ftrs, enable_branch_exp, l1, l2,
         x_include_mlp, x_include_orig, enable_token_supervision, model_type,
         mlp_n_hidden, mlp_n_layers, mlp_activation,
         use_loss_mask, wcn_aggreg,
         override_slots,
         templates_file

         ):

    # HACK:
    if early_stopping_group.startswith('req_'):
        early_stopping_group = "requested"

    output_dir = init_env(out)
    mon_train = TrainingStats()
    mon_valid = TrainingStats()
    mon_extreme_examples = TrainingStats()
    stats_obj = dict(
        train=mon_train.data,
        mon_extreme_examples=mon_extreme_examples.data,
        args=args_lst
    )

    logging.info('XTrack has been started.')
    try:
        logging.info('GIT rev: %s' % get_git_revision_hash())
    except:
        logging.info('GIT rev: unknown')
    logging.info('Output dir: %s' % output_dir)
    #logging.info('Initializing random seed to 0.')
    #random.seed(0)
    #np.random.seed(0)
    logging.info('Argv: %s' % str(sys.argv))
    logging.info('Effective args:')
    for arg_name, arg_value in args_lst:
        logging.info('    %s: %s' % (arg_name, arg_value))
    logging.info('Experiment path: %s' % experiment_path)

    if debug_theano:
        theano.config.mode = 'FAST_COMPILE'
        theano.config.optimizer = 'None'
        theano.config.linker = 'py'

    train_path = os.path.join(experiment_path, 'train.json')
    xtd_t = Data.load(train_path)

    valid_path = os.path.join(experiment_path, 'dev.json')
    xtd_v = Data.load(valid_path)

    test_path = os.path.join(experiment_path, 'test.json')
    xtd_s = Data.load(test_path)

    if templates_file:
        tpls = TemplateExtractor.load(templates_file)
        tpl_dialogs = tpls.sample_dialogs(ontology=xtd_t.ontology)
    else:
        tpl_dialogs = []

    slots = xtd_t.slots if not override_slots else override_slots.split(',')
    classes = xtd_t.classes
    class_groups = xtd_t.slot_groups
    shown_groups = []
    for group, group_slots in class_groups.iteritems():
        if set(group_slots).intersection(slots):
            shown_groups.append(group)

    for slot in slots:
        shown_groups.append(slot)
        class_groups[slot] = [slot]


    n_input_tokens = len(xtd_t.vocab)
    n_input_score_bins = len(xtd_t.score_bins)

    t = time.time()

    logging.info('Building model: %s' % model_type)
    if model_type == 'lstm':
        model = Model(slots=slots,
                      slot_classes=xtd_t.classes,
                      emb_size=emb_size,
                      no_train_emb=no_train_emb,
                      x_include_score=x_include_score,
                      x_include_token_ftrs=x_include_token_ftrs,
                      x_include_mlp=x_include_mlp,
                      x_include_orig=x_include_orig,
                      split_cost=split_cost,
                      wcn_aggreg=wcn_aggreg,
                      n_input_score_bins=n_input_score_bins,
                      n_cells=n_cells,
                      n_input_tokens=n_input_tokens,
                      oclf_n_hidden=oclf_n_hidden,
                      oclf_n_layers=oclf_n_layers,
                      oclf_activation=oclf_activation,
                      debug=debug,
                      rnn_n_layers=rnn_n_layers,
                      lstm_type=lstm_type,
                      lstm_update_thresh=lstm_update_thresh,
                      lstm_peepholes=lstm_peepholes,
                      lstm_bidi=lstm_bidi,
                      opt_type=opt_type,
                      momentum=momentum,
                      p_drop=p_drop,
                      init_emb_from=init_emb_from,
                      vocab=xtd_t.vocab,
                      vocab_ftr_map=xtd_t.vocab_ftrs,
                      ftr_emb_size=ftr_emb_size,
                      input_n_layers=input_n_layers,
                      input_n_hidden=input_n_hidden,
                      input_activation=input_activation,
                      token_features=token_features,
                      use_loss_mask=use_loss_mask,
                      enable_branch_exp=enable_branch_exp,
                      token_supervision=enable_token_supervision,
                      l1=l1,
                      l2=l2,
                      build_train=not debug_data
        )
    elif model_type == 'conv':
        model = SimpleConvModel(slots=slots,
                      slot_classes=xtd_t.classes,
                      emb_size=emb_size,
                      no_train_emb=no_train_emb,
                      x_include_score=x_include_score,
                      x_include_token_ftrs=x_include_token_ftrs,
                      x_include_mlp=x_include_mlp,
                      n_input_score_bins=n_input_score_bins,
                      n_cells=n_cells,
                      n_input_tokens=n_input_tokens,
                      oclf_n_hidden=oclf_n_hidden,
                      oclf_n_layers=oclf_n_layers,
                      oclf_activation=oclf_activation,
                      debug=debug,
                      rnn_n_layers=rnn_n_layers,
                      lstm_peepholes=lstm_peepholes,
                      lstm_bidi=lstm_bidi,
                      opt_type=opt_type,
                      momentum=momentum,
                      p_drop=p_drop,
                      init_emb_from=init_emb_from,
                      vocab=xtd_t.vocab,
                      input_n_layers=input_n_layers,
                      input_n_hidden=input_n_hidden,
                      input_activation=input_activation,
                      token_features=None,
                      enable_branch_exp=enable_branch_exp,
                      token_supervision=enable_token_supervision,
                      l1=l1,
                      l2=l2
        )
    elif model_type == 'baseline':
        model = BaselineModel(slots=slots,
                      slot_classes=xtd_t.classes,
                      oclf_n_hidden=oclf_n_hidden,
                      oclf_n_layers=oclf_n_layers,
                      oclf_activation=oclf_activation,
                      n_cells=n_cells,
                      debug=debug,
                      opt_type=opt_type,
                      momentum=momentum,
                      p_drop=p_drop,
                      vocab=xtd_t.vocab,
                      input_n_layers=input_n_layers,
                      input_n_hidden=input_n_hidden,
                      input_activation=input_activation,
                      token_features=None,
                      enable_branch_exp=enable_branch_exp,
                      token_supervision=enable_token_supervision,
                      l1=l1,
                      l2=l2
        )
    elif model_type == 'turn':
        model = TurnBasedModel(slots=slots,
                               slot_classes=xtd_t.classes,
                               opt_type=opt_type,
                               mlp_n_hidden=mlp_n_hidden,
                               mlp_n_layers=mlp_n_layers,
                               mlp_activation=mlp_activation,
                               debug=debug,
                               p_drop=p_drop,
                               vocab=xtd_t.vocab,
                               l1=l1,
                               l2=l2)
    elif model_type == 'turn_rnn':
        model = TurnBasedRNNModel(slots=slots,
                               slot_classes=xtd_t.classes,
                               opt_type=opt_type,
                               n_cells=n_cells,
                               mlp_n_hidden=mlp_n_hidden,
                               mlp_n_layers=mlp_n_layers,
                               mlp_activation=mlp_activation,
                               debug=debug,
                               p_drop=p_drop,
                               vocab=xtd_t.vocab,
                               l1=l1,
                               l2=l2)
    else:
        raise Exception()

    logging.info('Rebuilding took: %.1f' % (time.time() - t))

    if load_params:
        logging.info('Loading parameters from: %s' % load_params)
        model.load_params(load_params)

    tracker_valid = XTrack2DSTCTracker(xtd_v, [model], slots, override_groups=class_groups)
    tracker_test = XTrack2DSTCTracker(xtd_s, [model], slots, override_groups=class_groups)

    #model.visualize(xtd_v.sequences, slots)
    #return

    valid_data_y = model.prepare_data_train(xtd_v.sequences, slots, debug_data)
    #valid_data = model.prepare_data_predict(xtd_v.sequences, slots)
    if not eval_on_full_train:
        selected_train_seqs = []
        for i in range(100):
            ndx = random.randint(0, len(xtd_t.sequences) - 1)
            selected_train_seqs.append(xtd_t.sequences[ndx])
    else:
        selected_train_seqs = xtd_t.sequences

    #train_data = model.prepare_data_train(selected_train_seqs, slots)

    joint_slots = ['joint_%s' % str(grp) for grp in class_groups.keys()]
    #best_acc = {slot: 0 for slot in xtd_v.slots + joint_slots}
    #best_acc_train = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_tracking_acc = 0.0
    n_valid_not_increased = 0
    et = None

    def recreate_minibatches():
        seqs = list(xtd_t.sequences)
        seqs = seqs * mb_mult_data
        #random.shuffle(seqs)
        seqs.sort(key=lambda x: len(x['data']))
        minibatches = prepare_minibatches(seqs, mb_size, model, slots)
        minibatches = zip(itertools.count(), minibatches)
        mb_ids = range(len(minibatches))
        mb_to_go = []

        mb_ids += ['g'] * len(mb_ids)

        #for mb in minibatches:
        #    print mb[1][0].shape[0]

        return minibatches, mb_ids, mb_to_go

    minibatches, mb_ids, mb_to_go = recreate_minibatches()

    minibatches_valid = minibatches[:int(len(minibatches) * 1.0 / 20)]
    logging.info('We have %d minibatches.' % len(minibatches))

    tpl_generate = {
        'cache': [],
        'cntr': 0,
    }
    def generate_minibatch():
        if not tpl_generate['cache']:
            xtd_builder = DataBuilder(
                based_on=xtd_t,
                include_base_seqs=False,
                slots=slots,
                slot_groups=xtd_t.slot_groups,
                oov_ins_p=0.0,
                word_drop_p=0.0,
                include_system_utterances=True,
                nth_best=[0],
                score_bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
                ontology=xtd_t.ontology,
                debug_dir='/tmp/',
                concat_whole_nbest=False,
                include_whole_nbest=False,
                split_dialogs=False,
                sample_subdialogs=0,
                words=None,
                generate=None,
                tag_only=False,
                tagged=False,
                no_label_weight=True
            )
            seqs = xtd_builder.build(tpl_dialogs).sequences
            mbs = prepare_minibatches(seqs, mb_size, model, slots)

            tpl_generate['cache'].extend(mbs)
            tpl_generate['cntr'] = 0

        mb_id = tpl_generate['cntr']
        res = tpl_generate['cache'][:mb_size]
        del tpl_generate['cache'][:mb_size]
        tpl_generate['cntr'] += 1

        return mb_id, res

    example_cntr = 0
    timestep_cntr = 0
    stats = TrainingStats()

    def test_model(i):
        track_log_test = os.path.join(output_dir, 'track_log_test.%.10d.txt' % i)
        _, test_track_score = tracker_test.track(track_log_test)
        for group, accuracy in sorted(test_track_score.iteritems(),
                                  key=lambda (g, _): g):
            if not group in shown_groups:
                    continue
            logging.info('Test acc %15s: %10.2f %%'
                         % (group, accuracy * 100))


    epoch = 0
    last_es_epoch = 0

    #init_valid_loss = model._loss(*valid_data_y)
    #plot_loss(model, valid_data_y)
    #logging.info('Initial valid loss: %.10f' % init_valid_loss)

    if not valid_after:
        valid_after = len(xtd_t.sequences)

    if load_params:
        logging.info('Running test.')
        try:
            test_model(0)
        except Exception, e:
            logging.error('Testing the model failed: %s' % e)

    mb_loss = {}
    last_valid = 0
    last_inline_print = time.time()
    last_inline_print_cnt = 0
    best_track_acc = defaultdict(float)
    best_params = model.dump_params()
    while True:
        if len(mb_to_go) == 0:
            #logging.info('Recreating minibatches.')
            #minibatches, mb_ids, mb_to_go = recreate_minibatches()
            #logging.info('Batches recreated.')
            mb_to_go = list(mb_ids)
            epoch += 1

            #if n_valid_not_increased >= n_early_stopping:
            if epoch - last_es_epoch > n_early_stopping:
                lr = lr * 0.95 #/ lr_anneal_factor
                logging.info('New learning rate: %.5f' % lr)
                n_valid_not_increased = 0
                last_es_epoch = epoch
                model.push_params(best_params)

                try:
                    test_model(epoch)
                except Exception, e:
                    logging.error('Testing the model failed: %s' % e)

            if lr < 0.000001 or n_epochs > 0 and n_epochs < epoch:
                break

        mb_ndx = random.choice(mb_to_go)
        mb_to_go.remove(mb_ndx)

        #mb_id, mb_data = random.choice(minibatches)
        if mb_ndx == 'g':
            mb_id, mb_data = generate_minibatch()
        else:
            mb_id, mb_data = minibatches[mb_ndx]

        if train_noise:
            mb_data = add_train_noise(mb_data, train_noise, len(xtd_t.vocab))
        #if et is not None:
        #    epoch_time = time.time() - et
        #else:
        #    epoch_time = -1.0
        #logging.info('Epoch #%d (last epoch took %.1fs) (seen %d examples)' %
        #             (i, epoch_time, example_cntr ))

        #et = time.time()
        mb_done = 0
        t = time.time()
        _train = random.choice(model._train)
        (loss, update_ratio) = _train(lr, *mb_data)
        mb_loss[mb_ndx] = loss
        t = time.time() - t
        stats.insert(loss=loss, update_ratio=update_ratio, time=t)

        x = mb_data[0]
        example_cntr += x.shape[1]
        timestep_cntr += x.shape[0]
        mb_done += 1

        if time.time() - last_inline_print > 1.0:
            diff = time.time() - last_inline_print
            last_inline_print = time.time()
            progress = 1 - len(mb_to_go) * 1.0 / len(mb_ids)
            inline_print("     %3d%% %6d examples, %4d examples/s" % (
                progress * 100,
                example_cntr,
                (example_cntr - last_inline_print_cnt) / diff
            ))
            last_inline_print_cnt = example_cntr


        if (example_cntr - last_valid) >= valid_after:
            inline_print("")
            last_valid = example_cntr
            params_file = os.path.join(output_dir, 'params.%.3d.p' %
                                       epoch)
            logging.info('Saving parameters: %s' % params_file)
            model.save_params(params_file)

            #logging.info('Visualizing.')
            #model.visualize(xtd_v.sequences, slots)

            valid_loss = model._loss(*valid_data_y)
            update_ratio = stats.mean('update_ratio')
            update_ratio = stats.mean('update_ratio')

            logging.info('Valid # of examples: %d' % len(valid_data_y[-1]))
            track_log = os.path.join(output_dir, 'track_log.%.3d.txt' % epoch)
            track_result, track_score = tracker_valid.track(track_log)

            tracker_output = tracker_valid.prepare_dstc_format(0.0, "valid", track_result)
            params_file = os.path.join(output_dir, 'params.%.3d.p' %
                                       epoch)

            for group, accuracy in sorted(track_score.iteritems(),
                                          key=lambda (g, _): g):
                if not group in shown_groups:
                    continue

                logging.info('Valid acc %15s: %10.2f %%'
                             % (group, accuracy * 100))


                if group == early_stopping_group:
                    if accuracy <= best_track_acc[group]:
                        n_valid_not_increased += 1
                    else:
                        n_valid_not_increased = 0
                        best_params = model.dump_params()
                        logging.info('Noting the best params for later.')

                best_track_acc[group] = max(accuracy, best_track_acc[group])

            for group in sorted(track_score, key=lambda g: g):
                if not group in shown_groups:
                    continue

                logging.info('Best acc %15s:  %10.2f %%'
                             % (group, best_track_acc[group] * 100))
            logging.info('Train loss:         %10.2f' % stats.mean('loss'))
            logging.info('Mean update ratio:  %10.6f' % update_ratio)
            logging.info('Mean mb time:       %10.4f' % stats.mean('time'))
            logging.info('Epoch:              %10d (%d mb remain)' % (epoch,
                                                                     len(mb_to_go)))
            logging.info('Example:            %10d' % example_cntr)

            mon_train.insert(
                time=time.time(),
                example=example_cntr,
                timestep_cntr=timestep_cntr,
                mb_id=mb_id,
                train_loss=stats.mean('loss'),
                valid_loss=valid_loss,
                update_ratio=stats.mean('update_ratio'),
                tracking_acc=track_score
            )

            stats_path = os.path.join(output_dir, 'stats.json')
            with open(stats_path, 'w') as f_out:
                json.dump(stats_obj, f_out)
                os.system('ln -f -s "%s" "xtrack2_vis/stats.json"' %
                          os.path.join('..', stats_path))

            stats = TrainingStats()

    params_file = os.path.join(output_dir, 'params.final.p')
    logging.info('Saving final params to: %s' % params_file)
    model.save_params(params_file)

    return best_tracking_acc


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path')

    parser.add_argument('--eid', default='xtrack_experiment')
    parser.add_argument('--valid_after', default=None, type=int)

    # Experiment params.
    parser.add_argument('--load_params', default=None)
    parser.add_argument('--save_params', default=None)
    parser.add_argument('--out', default='/tmp/xtrack2_out')

    parser.add_argument('--model_type', default="lstm", type=str)

    # XTrack params.
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--n_early_stopping', default=10, type=int)
    parser.add_argument('--early_stopping_group', default="goals", type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_anneal_factor', default=2.0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--p_drop', default=0.0, type=float)
    parser.add_argument('--opt_type', default='sgd', type=str)
    parser.add_argument('--mb_size', default=1, type=int)
    parser.add_argument('--mb_mult_data', default=1, type=int)
    parser.add_argument('--l1', default=0.0, type=float)
    parser.add_argument('--l2', default=0.0, type=float)

    parser.add_argument('--n_cells', default=5, type=int)
    parser.add_argument('--emb_size', default=7, type=int)
    parser.add_argument('--ftr_emb_size', default=5, type=int)
    parser.add_argument('--wcn_aggreg', default='flatten')
    parser.add_argument('--x_include_score', default=False, action='store_true')
    parser.add_argument('--x_include_orig', default=False, action='store_true')
    parser.add_argument('--x_include_token_ftrs', default=False,
                        action='store_true')
    parser.add_argument('--x_include_mlp', default=False, action='store_true')
    parser.add_argument('--init_emb_from', default=None, type=str)
    parser.add_argument('--no_train_emb', default=False, action='store_true')
    parser.add_argument('--token_features', default=False,
                        action='store_true')
    parser.add_argument('--split_cost', default=False, action='store_true')
    parser.add_argument('--train_noise', default=0.0, type=float)

    parser.add_argument('--input_n_hidden', default=32, type=int)
    parser.add_argument('--input_n_layers', default=0, type=int)
    parser.add_argument('--input_activation', default="sigmoid", type=str)

    parser.add_argument('--mlp_n_hidden', default=32, type=int)
    parser.add_argument('--mlp_n_layers', default=0, type=int)
    parser.add_argument('--mlp_activation', default="tanh", type=str)

    parser.add_argument('--oclf_n_hidden', default=32, type=int)
    parser.add_argument('--oclf_n_layers', default=0, type=int)
    parser.add_argument('--oclf_activation', default="tanh", type=str)

    parser.add_argument('--rnn_n_layers', default=1, type=int)

    parser.add_argument('--lstm_type', default='vanilla')
    parser.add_argument('--lstm_update_thresh', default=0.0, type=float)
    parser.add_argument('--lstm_peepholes', default=False,
                        action='store_true')
    parser.add_argument('--lstm_bidi', default=False,
                        action='store_true')

    parser.add_argument('--use_loss_mask', default=False,
                        action='store_true')


    parser.add_argument('--debug', default=False,
                        action='store_true')
    parser.add_argument('--debug_data', default=False,
                        action='store_true')
    parser.add_argument('--debug_theano', default=False,
                        action='store_true')
    #parser.add_argument('--track_log', default='track_log.txt', type=str)
    #parser.add_argument('--track_log_test', default='track_log_test.txt', type=str)
    parser.add_argument('--eval_on_full_train', default=False,
                        action='store_true')

    parser.add_argument('--enable_branch_exp', default=False,
                        action='store_true')
    parser.add_argument('--enable_token_supervision', default=False,
                        action='store_true')

    parser.add_argument('--override_slots', default=None)

    parser.add_argument('--templates_file', default=None)

    return parser

if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    pdb_on_error()
    args_lst = list(sorted(vars(args).iteritems()))
    main(args_lst, **vars(args))
