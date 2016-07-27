import argparse
import json
import os

import operator


def main(
    in_dialogs_folder,
    in_chopped_dialogs_folder,
    in_dataset_name,
    in_scripts_config_folder,
    in_trackfiles_to_merge,
    in_result_trackfile
):
    original_sessions = load_sessions(
        in_dialogs_folder,
        in_dataset_name,
        in_scripts_config_folder
    )
    original_chopped_sessions = load_sessions(
        in_chopped_dialogs_folder,
        None,
        in_scripts_config_folder
    )
    chopped_tracks = []
    for trackfile_name in in_trackfiles_to_merge:
        with open(trackfile_name) as trackfile:
            chopped_tracks.append(json.load(trackfile))
    chopped_sessions = reduce(
        lambda x, y: x + y['sessions'],
        chopped_tracks,
        []
    )
    rebuild_utter_indices(
        chopped_sessions,
        original_chopped_sessions
    )
    merged_sessions = [
        rebuild_session(original_session, chopped_sessions)
        for original_session in original_sessions
    ]
    result_json = {
        'dataset': in_dataset_name,
        'wall_time': max(map(operator.itemgetter('wall_time'), chopped_tracks)),
        'sessions': merged_sessions
    }
    with open(in_result_trackfile, 'w') as out_file:
        json.dump(result_json, out_file)


def rebuild_session(in_original_session, in_chopped_sessions):
    original_session_id = str(in_original_session['session_id'])
    utterances = reduce(
        lambda x, y: x + y['utterances'],
        filter(
            lambda z: z['session_id'].startswith(original_session_id),
            in_chopped_sessions
        ),
        []
    )
    utterances_map = {
        utterance['utter_index']: utterance
        for utterance in utterances
    }
    result_session = {
        'session_id': int(original_session_id),
        'utterances': []
    }
    for utterance in in_original_session['utterances']:
        utter_index = utterance['utter_index']
        if utter_index in utterances_map:
            result_session['utterances'].append(utterances_map[utter_index])
        else:
            result_session['utterances'].append({'utter_index': utter_index})
    return result_session


def rebuild_utter_indices(in_modified_sessions, in_original_sessions):
    original_session_map = {
        session['session_id']: session
        for session in in_original_sessions
    }
    for session in in_modified_sessions:
        original_session = original_session_map[session['session_id']]
        for modified_utterance, original_utterance in zip(
            session['utterances'], original_session['utterances']
        ):
            modified_utterance['utter_index'] = \
                original_utterance['utter_index']


def load_sessions(in_dialogs_folder, in_dataset_name, in_scripts_config_folder):
    # all dirs in the data root folder
    dataset_files = os.walk(in_dialogs_folder).next()[1]
    if in_dataset_name:
        config_filename = os.path.join(
            in_scripts_config_folder,
            in_dataset_name + '.flist'
        )
        with open(config_filename) as config_file:
            dataset_files = [
                line.strip()
                for line in config_file
                if len(line.strip())
            ]
    dataset_dialogs = []
    for filename in dataset_files:
        dialog_filename = os.path.join(in_dialogs_folder, filename, 'log.json')
        with open(dialog_filename) as dialog_file:
            dataset_dialogs.append(json.load(dialog_file))
    return dataset_dialogs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogs_folder', required=True)
    parser.add_argument('--chopped_dialogs_folder', required=True)
    parser.add_argument(
        '--scripts_config_folder',
        default='dstc5_scripts/config'
    )
    parser.add_argument(
        '--dataset_name',
        default='dstc45_dev',
        help='"dstc<N>_train/dev/test"'
    )
    parser.add_argument(
        '--trackfiles_to_merge',
        default=','.join([
            'dstc45_dev_FOOD.out',
            'dstc45_dev_ATTRACTION.out',
            'dstc45_dev_TRANSPORTATION.out',
            'dstc45_dev_SHOPPING.out',
            'dstc45_dev_ACCOMMODATION.out'
        ]),
        help='"FOOD,ATTRACTION,..."'
    )
    parser.add_argument(
        '--result_trackfile',
        required=True
    )

    args = parser.parse_args()
    trackfiles_to_merge = [
        trackfile.strip()
        for trackfile in args.trackfiles_to_merge.split(',')
    ]

    main(
        args.dialogs_folder,
        args.chopped_dialogs_folder,
        args.dataset_name,
        args.scripts_config_folder,
        trackfiles_to_merge,
        args.result_trackfile
    )
