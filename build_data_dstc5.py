import data_utils
import xtrack2_config
from dstc5_scripts import ontology_reader
from import_dstc5 import ONTOLOGY_FILE, build_unified_ontology


def main(skip_dstc_import_step, builder_type):
    ONTOLOGY = build_unified_ontology(
        ontology_reader.OntologyReader(ONTOLOGY_FILE)
    )

    data_utils.prepare_experiment(
        experiment_name='e2_tagged_%s' % builder_type,
        data_directory=xtrack2_config.data_directory,
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
        builder_type=builder_type
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_dstc_import_step', action='store_true',
                        default=False)
    parser.add_argument('--builder_type', default='xtrack_dstc4')

    args = parser.parse_args()
    main(**vars(args))