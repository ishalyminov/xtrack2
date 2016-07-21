import argparse
import json
import os
import shutil

import re


def make_datasets_map(in_scripts_config_dir):
    datasets_map = {}
    for filename in os.listdir(in_scripts_config_dir):
        name, ext = os.path.splitext(filename)
        if ext != '.flist':
            continue
        with open(os.path.join(in_scripts_config_dir, filename)) as dataset:
            datasets_map[name] = [file.strip() for file in dataset.readlines()]
    return datasets_map


def translate_dialog_baseline(in_log, in_translations):
    translated_log = dict(in_log)
    for utterance, translations in zip(
            translated_log['utterances'],
            in_translations['utterances']
    ):
        top_hyp = ''
        align = []
        if len(translations['translated']):
            top_hyp = translations['translated'][0]['hyp']
            align = translations['translated'][0]['align']
        utterance['transcript'] = ' '.join(
            clean_tokens(segment_baseline(top_hyp, align))
        )
    return translated_log


def segment_baseline(in_phrase, in_align):
    result = []
    for eng_token, chars in in_align:
        result.append(''.join([in_phrase[char] for char in chars]))
    return result


def clean_tokens(in_tokens):
    result = map(
        lambda token: re.sub(r'[\W]+', '', token, flags=re.UNICODE),
        in_tokens
    )
    return result


def process_dialog(in_src_dir, in_dst_dir, in_dialog_name, in_translate_flag):
    dialog_folder = os.path.join(in_dst_dir, in_dialog_name)
    if os.path.isdir(dialog_folder):
        shutil.rmtree(dialog_folder)
    shutil.copytree(os.path.join(in_src_dir, in_dialog_name), dialog_folder)
    if in_translate_flag:
        translated_log = None
        with \
                open(os.path.join(dialog_folder, 'log.json')) as log_file, \
                open(os.path.join(dialog_folder, 'translations.json')) as translations_file:
            log = json.load(log_file)
            translations = json.load(translations_file)
            translated_log = translate_dialog_baseline(log, translations)
        with open(os.path.join(dialog_folder, 'log.json'), 'w') as log_file:
            json.dump(translated_log, log_file)


def main(in_input_dir, in_output_dir, in_scripts_config_dir):
    datasets_map = make_datasets_map(in_scripts_config_dir)

    for dataset, dialogs in datasets_map.items():
        translate = dataset.endswith('train')
        for dialog in dialogs:
            process_dialog(in_input_dir, in_output_dir, dialog, translate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scripts_config_dir', required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.scripts_config_dir)