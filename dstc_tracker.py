"""{
    "wall-time": 5.825495004653931,
    "dataset": "dstc2_dev",
    "sessions": [
        {
            "session-id": "voip-f246dfe0f2-20130328_161556",
            "turns": [
                {
                    "goal-labels": {
                        "pricerange": {
                            "expensive": 0.9883454175454712
                        },
                        "area": {
                            "south": 0.9673269337257503
                        }
                    },
                    "goal-labels-joint": [
                        {
                            "slots": {
                                "pricerange": "expensive",
                                "area": "south"
                            },
                            "score": 0.9777797002475338
                        }
                    ],
                    "method-label": {
                        "byconstraints": 0.9999999999999999
                    },
                    "requested-slots": {}
                }
        }
}
"""

import collections
import time
import json
import logging
import numpy as np
import argparse

from data import Data, Tagger
from dstc5_scripts import ontology_reader
from utils import pdb_on_error
from model import Model
from model_baseline import BaselineModel


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
    def __init__(self, data, models, ontology):
        assert len(models), 'You need to specify some models.'

        self.data = data
        self.models = models
        self.main_model = models[0]
        self.ontology = ontology

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

    def build_output(self, pred, label, segment_id):
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

        goal_labels = {
            slot: [pred[slot]]
            for slot in self.data.slots
            if pred[slot] != self.data.null_class \
            # and slot in self.ontology.tagsets.get(segment_id, {}).keys() \
            # and pred[slot] in self.ontology.tagsets.get(segment_id, {}).get(slot, [])
        }

        return {
            'frame_label': goal_labels,
        }, goals_correct

    def _label_empty(self, lbl):
        res = True
        for val in lbl.values():
            res &= val == 0
        return res

    def _make_model_predictions(self, data):
        preds = []
        for model in self.models:
            pred = model._predict(*data)
            preds.append(pred)
        return preds

    def track(self, tracking_log_file_name=None, output_len_accuracy=False):
        data = self.main_model.prepare_data_predict(
            self.data.sequences,
            self.data.slots
        )

        preds = self._make_model_predictions(data)

        pred = []
        for slot_preds in zip(*preds):
            slot_res = np.array(slot_preds[0])
            for slot_pred in slot_preds[1:]:
                slot_res += slot_pred
            pred.append(slot_res / len(slot_preds))

        pred_ptr = 0

        len_accuracy = \
            collections.defaultdict(lambda: collections.defaultdict(int))
        len_accuracy_n = \
            collections.defaultdict(lambda: collections.defaultdict(int))
        accuracy = collections.defaultdict(int)
        accuracy_n = collections.defaultdict(int)
        result = []
        if tracking_log_file_name:
            self.track_log = open(tracking_log_file_name, 'w')
        else:
            self.track_log = open('/dev/null', 'w')

        for dialog in self.data.sequences:
            self.track_log.write(">> Dialog: %s\n" % dialog['id'])
            self.track_log.write("\n")
            turns = []
            last_pos = 0
            state_component_mentioned = False
            for utter_index, lbl in enumerate(dialog['labels']):
                last_pos = lbl['time'] + 1
                segment_id, segment_bio = lbl['segment_id'], lbl['segment_bio']

                out, goals_correct = self.build_output(
                    [
                        pred[i][pred_ptr]
                        for i, _ in enumerate(self.data.slots)
                    ],
                    lbl['slots'],
                    segment_id
                )
                if dialog['tags']:
                    dialog['tags'] = self._replace_tags(out, dialog['tags'])
                #self.track_log.write(json.dumps(out))
                #self.track_log.write("\n")
                if segment_bio == 'O':
                    del out['frame_label']
                out['utter_index'] = utter_index
                turns.append(out)
                pred_ptr += 1

                if not self._label_empty(lbl['slots']) or state_component_mentioned:
                    state_component_mentioned = True

                    for group, slots in self.slot_groups.iteritems():
                        if goals_correct[group]:
                            accuracy[group] += 1
                            len_accuracy[last_pos][group] += 1
                        accuracy_n[group] += 1
                        len_accuracy_n[last_pos][group] += 1

            result.append({
                'session_id': dialog['id'],
                'utterances': turns
            })

            #self.track_log.write("\n")

        if len(pred[0]) != pred_ptr:
            raise Exception('Data mismatch.')

        for group in self.slot_groups:
            accuracy[group] = accuracy[group] * 1.0 / max(1, accuracy_n[group])
            for t in len_accuracy:
                factor = 1.0 / max(1, len_accuracy_n[t][group])
                len_accuracy[t][group] = len_accuracy[t][group] * factor

        res = [result, accuracy]
        if output_len_accuracy:
            res.append(len_accuracy)
            res.append(len_accuracy_n)
        return tuple(res)

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


def main(
        dataset_name, data_file, output_file,
        params_file, model_type, ontology
):
    models = []
    for pf in params_file:
        logging.info('Loading model from: %s' % pf)
        if model_type == 'lstm':
            model_cls = Model
        elif model_type == 'baseline':
            model_cls = BaselineModel
        models.append(model_cls.load(pf, build_train=False))

    logging.info('Loading data: %s' % data_file)
    data = Data.load(data_file)

    logging.info('Starting tracking.')
    tracker = XTrack2DSTCTracker(
        data,
        models,
        ontology_reader.OntologyReader(ontology)
    )

    t = time.time()
    result, tracking_accuracy, len_accuracy, len_accuracy_n = \
        tracker.track(output_len_accuracy=True)
    t = time.time() - t
    logging.info('Tracking took: %.1fs' % t)
    for group, accuracy in tracking_accuracy.iteritems():
        logging.info('Accuracy %s: %.2f %%' % (group, accuracy * 100))
        for t in len_accuracy:
            print '%d %.2f %d' % (
                t, len_accuracy[t][group], len_accuracy_n[t][group]
            )


    tracker_output = {
        'wall_time': float(t),
        'dataset': dataset_name,
        'sessions': result
    }

    logging.info('Writing to: %s' % output_file)
    with open(output_file, 'w') as f_out:
        json.dump(tracker_output, f_out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='test')
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--params_file', action='append', required=True)
    parser.add_argument('--model_type', default='lstm'),
    parser.add_argument('--ontology', required=True)

    pdb_on_error()
    init_logging()
    main(**vars(parser.parse_args()))
