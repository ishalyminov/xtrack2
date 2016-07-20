#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0111

import argparse
import json
import os.path
import sys

import re


NEAR_INF = sys.float_info.max


class DSTC5ParsedDialog(object):
    """Dialog log.

    Representation of one dialog.

    Attributes:
        turns: A list of dialog turns.
        session_id: ID of the dialog.
    """

    def __init__(self, log, labels):
        """Initialises a dialogue object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """

        self.utterances = []
        self.markup = {
            key: value
            for key, value in log.iteritems()
            if key != 'utterances'
        }
        self.session_id = self.markup['session_id']

        if labels:
            for utterance, label in zip(log['utterances'], labels['utterances']):
                self.utterances.append(Turn(utterance, label))
        else:
            for turn_json in log['utterances']:
                self.turns.append(Turn(turn_json, None))

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Dialog:\n'
        repr_str += (indent + step) * ' ' + 'id: "%s",\n' % self.session_id
        repr_str += (indent + step) * ' ' + 'turns:\n'
        for turn in self.turns:
            repr_str += turn.pretty_print(indent + 2 * step, step) + '\n'
        return repr_str

    def __str__(self):
        return self.pretty_print()

    def __repr__(self):
        return 'Dialog(id="%s")' % self.session_id


class Turn(object):
    """
        One turn of a dialog.
        Representation of one turn in a dialog. Contains information about
        things the user asked as well as the reply from dialog manager.
    """

    def __init__(self, turn, labels):
        """Initialises a turn object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.utter_index = turn['utter_index']
        self.transcript = ''
        self.segment_bio = turn['segment_info']['target_bio']
        self.segment_topic = turn['segment_info']['topic']
        self.speaker = turn['speaker']
        self.transcript = turn.get('transcript', None)
        self.slots_map = {}
        if labels:
            full_tagged_utterance = ' '.join(labels['semantic_tagged'])
            self.slots_map = parse_semantic_tagged_utterance(
                full_tagged_utterance
            )
            self.dialog_state = parse_dstc5_dialog_state(
                labels.get('frame_label', {})
            )


def parse_dstc5_dialog_state(in_dialog_state):
    return {
        key: '___'.join(sorted(value_list[-2:]))
        for key, value_list in in_dialog_state.iteritems()
    }


def parse_semantic_tagged_utterance(in_utterance):
    result = []
    for slot_name, slot_value in re.findall('<(.+?)>(.+?)</.+?>', in_utterance):
        slot_map = {'value': slot_value}
        name_attrs = slot_name.split(' ')
        slot_map['name'] = name_attrs[0]
        slot_map['attrs'] = {}
        for attr in name_attrs[1:]:
            attr_name, attr_value = attr.replace('"', '').split('=')
            slot_map['attrs'][attr_name] = attr_value
        result.append(slot_map)
    return result


def parse_dialog_from_directory(dialog_dir):
    """
    Keyword arguments:
        dialog_dir: the directory immediately containing the dialogue JSON logs
        regress_to_dais: whether to regress DA scores to scores of single DAIs
        norm_slu_scores: whether scores for SLU hypotheses should be
                         normalised to the scale [0, 1]
        slot_normaliser: instance of a normaliser with normalise method
        reranker_model: if given, an SLU reranker will be applied, using the
                        trained model whose file name is passed in this
                        argument
    """
    log = json.load(open(os.path.join(dialog_dir, 'log.json')))

    labels_file_name = os.path.join(dialog_dir, 'label.json')
    if os.path.exists(labels_file_name):
        labels = json.load(open(labels_file_name))
    else:
        labels = None

    d = DSTC5ParsedDialog(log, labels)

    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load dstc data.")
    parser.add_argument(
        '-d',
        required=True,
        nargs="+",
        dest='dirs',
        metavar="dir",
        help="Directories with logs."
    )
    args = parser.parse_args()

    dialogs = [
        parse_dialog_from_directory(directory)
        for directory in args.dirs
    ]
