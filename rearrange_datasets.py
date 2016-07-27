import argparse
import json
import os

# used for filtering off too short dialogs
import random

MINIMUM_DIALOG_SEGMENT_LENGTH = 2

EVEN_DATASET_SIZE_RATIOS = {
    'train': 0.7,
    'dev': 0.1,
    'test': 0.2
}


def main(
    in_dialogs_folder,
    in_dataset_names,
    in_scripts_config_folder,
    in_topics
):
    rearrange_datasets(
        in_dialogs_folder,
        in_scripts_config_folder,
        in_dataset_names,
        in_topics
    )


def rearrange_datasets(
    in_data_dir,
    in_scripts_config_folder,
    in_dataset_names,
    in_topics
):
    for topic in in_topics:
        setup_by_topic = []
        for dataset in in_dataset_names:
            dataset_filename = os.path.join(
                in_scripts_config_folder,
                '{}_{}.flist'.format(dataset, topic)
            )
            with open(dataset_filename) as input:
                setup_by_topic += [line.strip() for line in input]
        setup_by_topic = filter_empty_dialogs(
            in_data_dir,
            setup_by_topic
        )
        topic_dialogs = setup_by_topic
        random.shuffle(topic_dialogs)

        used_dialogs_number = 0
        for dataset in in_dataset_names:
            dataset_filename = os.path.join(
                in_scripts_config_folder,
                '{}_{}.flist'.format(dataset, topic)
            )
            print 'Rearranging dataset ' + dataset_filename
            if not topic_dialogs:
                print 'TOPIC DIALOGS EMPTY!!!'
            dataset_type = dataset.split('_')[-1]
            dataset_size = \
               int(EVEN_DATASET_SIZE_RATIOS[dataset_type] * len(topic_dialogs))
            dataset_dialogs = topic_dialogs[
                used_dialogs_number:used_dialogs_number + dataset_size
            ]
            used_dialogs_number += dataset_size
            with open(dataset_filename, 'w') as output:
                print >>output, '\n'.join(dataset_dialogs)


def filter_empty_dialogs(in_data_dir, in_dialog_flist):
    result = []
    for dialog in in_dialog_flist:
        dialog_log = os.path.join(in_data_dir, dialog, 'log.json')
        with open(dialog_log) as input:
            dialog_json = json.load(input)
            if MINIMUM_DIALOG_SEGMENT_LENGTH <= len(dialog_json['utterances']):
                result.append(dialog)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogs_folder', required=True)
    parser.add_argument(
        '--scripts_config_folder',
        default='dstc5_scripts/config/chopped'
    )
    parser.add_argument(
        '--dataset_names',
        default='dstc45_train,dstc45_dev,dstc45_test',
        help='"train,dev..."'
    )
    parser.add_argument(
        '--topics',
        default='FOOD,ATTRACTION,TRANSPORTATION,SHOPPING,ACCOMMODATION',
        help='"FOOD,ATTRACTION,..."'
    )

    args = parser.parse_args()
    datasets = [dataset.strip() for dataset in args.dataset_names.split(',')]
    topics = [topic.strip() for topic in args.topics.split(',')]

    main(
        args.dialogs_folder,
        datasets,
        args.scripts_config_folder,
        topics,
    )
