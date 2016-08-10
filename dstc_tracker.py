import time
import json
import logging
import numpy as np
import argparse
from itertools import count

from data import Data, Tagger
from utils import pdb_on_error
from model import Model
from model_baseline import BaselineModelKeras
from model_simple_conv import SimpleConvModel

from dstc5_scripts.stat_classes import (
    Stat_Accuracy, Stat_Frame_Precision_Recall
)


def init_logging():
    # Setup logging.
    logger = logging.getLogger('XTrack')
    logger.setLevel(logging.DEBUG)

    logging_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(logging_format)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.root = logger


class XTrack2DSTCTracker(object):
    def __init__(self, data, models):
        assert len(models), 'You need to specify some models.'

        self.data = data
        self.models = models
        self.main_model = models[0]

        self.classes_rev = {}
        for slot in self.data.slots:
            self.classes_rev[slot] = {
                val: key
                for key, val in self.data.classes[slot].iteritems()
            }

        self.slot_groups = data.slot_groups
        self.tagger = Tagger()

    def _label_id_to_str(self, label):
        res = {}
        for slot in self.data.slots:
            res[slot] = self.classes_rev[slot][label[slot]]
        return res

    def build_output(
        self,
        pred,
        label,
        segment_id,
        in_accuracy_stat,
        in_frame_precision_recall_stat
    ):
        raw_labels = {}
        raw_label_probs = {}
        for i, slot in enumerate(self.data.slots):
            val = np.argmax(pred[i])
            val_prob = pred[i][val]
            if pred[i][val] == 0.0:
                val = 0
            raw_labels[slot] = val
            raw_label_probs[slot] = val_prob

        lbl = self._label_id_to_str(label)
        pred = self._label_id_to_str(raw_labels)
        for slot in self.data.slots:
            self.track_log.write(
                "  %s lbl(%s) pred(%s)\n" % (
                    slot,
                    lbl[slot],
                    pred[slot]
                )
            )
        goals_correct = {}
        for group, slots in self.slot_groups.iteritems():
            goals_correct[group] = True
            for i, slot in enumerate(slots):
                goals_correct[group] &= raw_labels[slot] == label[slot]

        label_list = {key: [value] for key, value in label.items()}
        raw_labels_list = {key: [value] for key, value in raw_labels.items()}
        in_accuracy_stat.add(label_list, raw_labels_list)
        in_frame_precision_recall_stat.add(label_list, raw_labels_list)

        goal_labels = {
            slot: pred[slot]
            for slot in self.data.slots
            if pred[slot] != self.data.null_class \
        }

        tracker_output = {
            'frame_label': goal_labels
        }
        return tracker_output, goals_correct

    def _label_empty(self, lbl):
        res = True
        for val in lbl.values():
            res &= val == 0
        return res

    def track(self, tracking_log_file_name=None):
        accuracy_stat = Stat_Accuracy()
        frame_precision_recall_stat = Stat_Frame_Precision_Recall()
        X = self.main_model.prepare_data_test(
            self.data.sequences,
            self.data.slots,
            with_labels=False
        )[0]

        prediction = self.main_model.predict(X)
        frame_labels = \
            self.main_model.get_frame_label_from_prediction(prediction)
        result = []
        if tracking_log_file_name:
            self.track_log = open(tracking_log_file_name, 'w')
        else:
            self.track_log = open('/dev/null', 'w')

        for dialog, frame_label in zip(self.data.sequences, frame_labels):
            print >>self.track_log, '>> Dialog: {}\n'.format(dialog['id'])
            turns = []
            for utter_index, lbl, in zip(
                count(),
                dialog['labels']
            ):
                last_pos = lbl['time'] + 1
                segment_id, segment_bio = lbl['segment_id'], lbl['segment_bio']

                out = {'frame_label': dict(frame_label)}
                if dialog['tags']:
                    dialog['tags'] = \
                        self._replace_tags(out, dialog['tags'])
                if segment_bio == 'O':
                    del out['frame_label']
                else:
                    out['frame_label'] = \
                        self._denormalize_frame_label(out['frame_label'])
                lbl_string = self._denormalize_frame_label(
                    self._slot_features_to_strings(lbl['slots'])
                )
                accuracy_stat.add(
                    out['frame_label'],
                    lbl_string
                )
                frame_precision_recall_stat.add(
                    out['frame_label'],
                    lbl_string
                )
                out['utter_index'] = utter_index
                turns.append(out)

            result.append({
                'session_id': dialog['id'],
                'utterances': turns
            })

        stats = {
            'accuracy': accuracy_stat.results()[0][2],
            'frame_precision': frame_precision_recall_stat.results()[0][2],
            'frame_recall': frame_precision_recall_stat.results()[1][2],
            'frame_f1': frame_precision_recall_stat.results()[2][2]
        }
        res = [result, stats]
        return tuple(res)

    def _slot_features_to_strings(self, in_frame_label):
        return {
            slot: self.main_model.slot_classes_reversed[slot][feature]
            for slot, feature in in_frame_label.items()
        }

    def _denormalize_frame_label(self, in_frame_label):
        new_frame_label = {}
        for slot, value in in_frame_label.items():
            denormalized_value = value.split('___')
            denormalized_value = [
                atomic_value.replace('_', ' ')
                for atomic_value in denormalized_value
            ]
            new_frame_label[slot] = denormalized_value
        return new_frame_label

    def _replace_tags(self, out, tags):
        new_tags = {}
        for slot, values in out['frame_label'].iteritems():
            new_tags[slot] = self._replace_tags_for_slot(slot, tags, values)
            return new_tags

    def _replace_tags_for_slot(self, slot, tags, values):
        new_res = []
        for slot_val in values:
            if slot_val.startswith('#%s' % slot):
                tag_id = int(slot_val.replace('#%s' % slot, ''))
                try:
                    tag_list = tags.get(slot, [])
                    tag_val = tag_list[tag_id]
                    tag_val = self.tagger.denormalize_slot_value(tag_val)
                    new_res.append(tag_val)
                except IndexError:
                    # This happens when the we predict a tag that
                    # does not exist.
                    new_res.append('_null_')
            else:
                new_res.append(slot_val)
        return new_res


def main(dataset, data_file, output_file, params_file, model_type):
    models = []
    for pf in params_file:
        logging.info('Loading model from: %s' % pf)
        if model_type == 'lstm':
            model_cls = Model
        elif model_type == 'simple_conv':
            model_cls = SimpleConvModel
        elif model_type == 'baseline':
            model_cls = BaselineModelKeras
        models.append(model_cls.load(pf))

    logging.info('Loading data: %s' % data_file)
    data = Data.load(data_file)

    logging.info('Starting tracking.')
    tracker = XTrack2DSTCTracker(data, models)

    t = time.time()
    result, stats = tracker.track()
    t = time.time() - t
    logging.info('Tracking took: %.1fs' % t)
    logging.info('Tracking stats: ')
    for metric, value in stats.items():
        if not value:
            value = 0.0
        logging.info('%s: %.5f %%' % (metric, value * 100))

    tracker_output = {
        'wall_time': float(t),
        'dataset': dataset,
        'sessions': result
    }

    logging.info('Writing to: %s' % output_file)
    with open(output_file, 'w') as f_out:
        json.dump(tracker_output, f_out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--params_file', action='append', required=True)
    parser.add_argument('--model_type', default='baseline'),

    pdb_on_error()
    init_logging()
    main(**vars(parser.parse_args()))
