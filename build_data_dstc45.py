import argparse
import os

import data_utils
from import_dstc45 import build_ontology_for_topic
from xtrack2_config import dstc45_ontology_filename
from dstc5_scripts import ontology_reader


def main(
    dialogs_folder,
    skip_dstc_import_step,
    builder_type,
    challenge_id,
    dataset_names,
    vocabulary_size_limit
):
    ontology = ontology_reader.OntologyReader(dstc45_ontology_filename)
    datasets = [dataset.strip() for dataset in dataset_names.split(',')]
    for topic in ontology.get_topics():
        datasets_for_topic = [
            ('{}_{}'.format(challenge_id, dataset), dataset)
            for dataset in datasets
        ]
        slots = ontology.get_slots(topic)
        data_utils.prepare_experiment(
            experiment_name='e2_tagged_%s_%s' % (challenge_id, topic),
            data_directory=os.path.join(dialogs_folder, topic),
            slots=slots,
            slot_groups={slot: [slot] for slot in slots},
            ontology=build_ontology_for_topic(ontology, topic),
            builder_opts={
                'tagged': True,
                'no_label_weight': True,
                'limit_vocabulary_size': vocabulary_size_limit
            },
            skip_dstc_import_step=skip_dstc_import_step,
            builder_type=builder_type,
            in_datasets=datasets_for_topic
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogs_folder', required=True)
    parser.add_argument(
        '--skip_dstc_import_step',
        action='store_true',
        default=False
    )
    parser.add_argument('--builder_type', default='xtrack_dstc45')
    parser.add_argument('--challenge_id', default='dstc45')
    parser.add_argument(
        '--dataset_names',
        default='train,dev',
        help='"name1,name2..."'
    )
    parser.add_argument('--vocabulary_size_limit', default=0, type=int)

    args = parser.parse_args()
    main(**vars(args))
