import argparse

import data_utils
from xtrack2_config import dstc45_ontology_filename, data_directory
from dstc5_scripts import ontology_reader
from import_dstc45 import build_unified_ontology


def main(skip_dstc_import_step, builder_type, dataset_names):
    ONTOLOGY = build_unified_ontology(
        ontology_reader.OntologyReader(dstc45_ontology_filename)
    )

    dataset_names = [dataset.strip() for dataset in dataset_names.split(',')]
    data_utils.prepare_experiment(
        experiment_name='e2_tagged_%s' % builder_type,
        data_directory=data_directory,
        slots=ONTOLOGY.keys(),
        slot_groups= {
            slot_name: [slot_name]
            for slot_name in ONTOLOGY.keys()
        },
        ontology=ONTOLOGY,
        builder_opts=dict(
            tagged=True,
            no_label_weight=True
        ),
        skip_dstc_import_step=skip_dstc_import_step,
        builder_type=builder_type,
        in_datasets=dataset_names
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip_dstc_import_step',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--builder_type',
        default='xtrack_dstc45'
    )
    parser.add_argument(
        '--dataset_names',
        default='train,dev,test',
        help='"name1,name2..."'
    )

    args = parser.parse_args()
    main(**vars(args))
