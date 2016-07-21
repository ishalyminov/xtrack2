import argparse
import codecs
import json
import os
import random
import shutil
import itertools
import collections

DIALOG_FILES = ['log.json', 'label.json', 'translations.json']
DIALOG_LEN_MIN = 20
DIALOG_LEN_MAX = 60


def chop_dialogs(in_src_folder, in_dst_folder, in_dialogs_to_process):
    root, dirs, files = os.walk(in_src_folder).next()
    chopped_dialogs_map = collections.defaultdict(lambda: [])
    for dirname in dirs:
        if dirname not in in_dialogs_to_process:
            continue
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


def chop_dataset_configs(
    in_chopped_dialogs_map,
    in_dataset_folder,
    in_dataset_names
):
    result_folder = os.path.join(in_dataset_folder, 'chopped')
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)
    for config_file in os.listdir(in_dataset_folder):
        name, ext = os.path.splitext(config_file)
        if ext != '.flist' or name not in in_dataset_names:
            continue
        modified_config = []
        with open(os.path.join(in_dataset_folder, config_file)) as input:
            for line in input:
                # removing leading zeros
                line = int(line.lstrip('0'))
                modified_config += in_chopped_dialogs_map[line]
        with open(os.path.join(result_folder, config_file), 'w') as output:
            print >>output, '\n'.join(modified_config)


def get_blank_dialog(in_log, in_label, in_translations):
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

    result = {
        'log': dict(log_header),
        'label': dict(label_header),
        'translations': dict(translations_header)
    }
    return result


def set_session_id(in_blank_dialog, in_session_id):
    for key in in_blank_dialog.keys():
        in_blank_dialog[key]['utterances'] = []
        in_blank_dialog[key]['session_id'] = in_session_id


def chop_dialog(
    in_log,
    in_label,
    in_translations,
    keep_segments=True,
    filter_general_talk=True
):
    get_dialog_length = lambda: \
        999999 if keep_segments \
        else int(
            DIALOG_LEN_MIN + random.random() * (DIALOG_LEN_MAX - DIALOG_LEN_MIN)
        )
    begin_new_dialog = lambda turn: \
        turn['segment_info']['target_bio'] == 'B' if keep_segments \
        else turns_in_current_dialog == current_dialog_length

    blank_dialog = get_blank_dialog(in_log, in_label, in_translations)

    result = []
    current_dialog_length = 0
    turns_in_current_dialog = 0

    for turn_index, log_turn, label_turn, translation_turn in zip(
        itertools.count(),
        in_log['utterances'],
        in_label['utterances'],
        in_translations['utterances']
    ):
        if filter_general_talk and log_turn['segment_info']['target_bio'] == 'O':
            continue
        if begin_new_dialog(log_turn):
            current_dialog_length = get_dialog_length()
            turns_in_current_dialog = 0
            new_dialog = get_blank_dialog(
                in_log,
                in_label,
                in_translations
            )
            set_session_id(
                new_dialog,
                '{}_{}'.format(in_log['session_id'], len(result))
            )
            result.append(new_dialog)
        result[-1]['log']['utterances'].append(log_turn)
        result[-1]['label']['utterances'].append(label_turn)
        result[-1]['translations']['utterances'].append(translation_turn)
        turns_in_current_dialog += 1
    return result


def load_dataset_info(in_dataset_names, in_scripts_config_folder):
    datasets = {}
    for name in in_dataset_names:
        dataset_filename = os.path.join(in_scripts_config_folder, name)
        with open(dataset_filename + '.flist') as dataset_file:
            datasets[name] = map(lambda x: x.strip(), dataset_file.readlines())
    return datasets


def main(
    in_dialogs_folder,
    in_output_folder,
    in_dataset_names,
    in_scripts_config_folder
):
    if os.path.isdir(in_output_folder):
        shutil.rmtree(in_output_folder)
        os.makedirs(in_output_folder)

    datasets = load_dataset_info(in_dataset_names, in_scripts_config_folder)

    chopped_dialogs_map = chop_dialogs(
        in_dialogs_folder,
        in_output_folder,
        reduce(lambda x, y: x + y, datasets.values(), [])
    )
    chop_dataset_configs(
        chopped_dialogs_map,
        in_scripts_config_folder,
        in_dataset_names
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogs_folder', required=True)
    parser.add_argument(
        '--scripts_config_folder',
        default='dstc5_scripts/config'
    )
    parser.add_argument('--output_folder', default='data/dstc5_chopped')
    parser.add_argument(
        '--dataset_names',
        default='dstc5_train,dstc5_dev',
        help='"train,dev..."'
    )

    args = parser.parse_args()

    main(
        args.dialogs_folder,
        args.output_folder,
        args.dataset_names.split(','),
        args.scripts_config_folder
    )
