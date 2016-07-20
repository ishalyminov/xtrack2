import argparse
import codecs
import json
import os
import random
import shutil
import sys

import itertools

import collections

DIALOG_FILES = ['log.json', 'label.json', 'translations.json']
DIALOG_LEN_MIN = 10
DIALOG_LEN_MAX = 30


def chop_dialogs(in_src_folder, in_dst_folder):
    root, dirs, files = os.walk(in_src_folder).next()
    chopped_dialogs_map = collections.defaultdict(lambda: [])
    for dirname in dirs:
        log, label, translations = map(
            lambda suffix: json.load(
                codecs.getreader('utf-8')(
                    open(os.path.join(root, dirname, suffix))
                )
            ),
            DIALOG_FILES
        )
        original_dialog_id = log['session_id']
        chopped_dialogs = chop_dialog(log, label, translations)
        for dialog in chopped_dialogs:
            chopped_dialog_id = dialog['log']['session_id']
            chopped_dialogs_map[original_dialog_id].append(chopped_dialog_id)
            dialog_folder = os.path.join(in_dst_folder, chopped_dialog_id)
            os.makedirs(dialog_folder)
            for dialog_file in DIALOG_FILES:
                file_name = os.path.join(dialog_folder, dialog_file)
                with codecs.getwriter('utf-8')(open(file_name, 'w')) as f_out:
                    json.dump(
                        dialog[dialog_file.split('.')[0]],
                        f_out,
                        ensure_ascii=False,
                        indent=4
                    )
    return chopped_dialogs_map


def chop_dataset_configs(in_chopped_dialogs_map, in_dataset_folder):
    result_folder = os.path.join(in_dataset_folder, 'chopped')
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)
    for config_file in os.listdir(in_dataset_folder):
        if not config_file.endswith('flist'):
            continue
        modified_config = []
        with open(os.path.join(in_dataset_folder, config_file)) as input:
            for line in input:
                # removing leading zeros
                line = int(line.lstrip('0'))
                modified_config += in_chopped_dialogs_map[line]
        with open(os.path.join(result_folder, config_file), 'w') as output:
            print >>output, '\n'.join(modified_config)


def chop_dialog(in_log, in_label, in_translations):
    def get_blank_dialog(in_session_id):
        result = {
            'log': dict(log_header),
            'label': dict(label_header),
            'translations': dict(translations_header)
        }
        for key in result.keys():
            result[key]['utterances'] = []
            result[key]['session_id'] = in_session_id
        return result

    get_dialog_length = lambda: int(
        DIALOG_LEN_MIN + random.random() * (DIALOG_LEN_MAX - DIALOG_LEN_MIN)
    )

    log_header = {
        key: in_log[key]
        for key in set(in_log.keys()).difference(['utterances'])
    }
    label_header = {
        key: in_label[key]
        for key in set(in_label.keys()).difference(['utterances'])
    }
    translations_header = {
        key: in_translations[key]
        for key in set(in_translations.keys()).difference(['utterances'])
    }

    result = []
    current_dialog_length = 0
    turns_in_current_dialog = 0

    for turn_index, log_turn, label_turn, translation_turn in zip(
        itertools.count(),
        in_log['utterances'],
        in_label['utterances'],
        in_translations['utterances']
    ):
        if turns_in_current_dialog == current_dialog_length:
            current_dialog_length = get_dialog_length()
            turns_in_current_dialog = 0
            result.append(
                get_blank_dialog(
                    '{}_{}'.format(log_header['session_id'], len(result))
                )
            )
        result[-1]['log']['utterances'].append(log_turn)
        result[-1]['label']['utterances'].append(label_turn)
        result[-1]['translations']['utterances'].append(translation_turn)
        turns_in_current_dialog += 1
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogs_folder', default='data/dstc5')
    parser.add_argument(
        '--scripts_config_folder',
        default='dstc5_scripts/config'
    )
    parser.add_argument(
        '--output_folder_folder',
        default='data/dstc5_chopped'
    )
    parser.add_argument(
        '--dataset_names',
        default='train,dev',
        help='"train,dev..."'
    )

    args = parser.parse_args()

    if os.path.isdir(args.output_folder):
        shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)
    chopped_dialogs_map = chop_dialogs(sys.argv[1], sys.argv[3])
    chop_dataset_configs(chopped_dialogs_map, sys.argv[2])
