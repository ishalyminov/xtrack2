import argparse
import os
import collections

import dstc_util
from xtrack2_config import dstc45_ontology_filename
from data_model import Dialog

from dstc5_scripts import ontology_reader

ONTOLOGY = {}


def build_unified_ontology(in_ontology_reader):
    merged_ontology = collections.defaultdict(lambda: set([]))
    ontologies = in_ontology_reader.get_tagsets()
    for topic_name, topic_ontology in ontologies.iteritems():
        for slot, slot_values in topic_ontology.iteritems():
            merged_ontology[slot].update(slot_values)
    result = {
        slot_name: list(slot_values)
        for slot_name, slot_values in merged_ontology.iteritems()
    }
    return result


def _stringify_act(in_slots_map):
    res = []
    for slot in in_slots_map:
        res.append(slot['name'])
        for name, value in slot['attrs'].iteritems():
            res.append(name.replace(' ', '_'))
            res.append(value.replace(' ', '_'))
    if len(res) == 0:
        res = ["sys"]
    return " ".join(res)


def import_dstc(data_dir, out_dir, flist, use_stringified_system_acts):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dialog_dirs = []
    with open(flist) as f_in:
        for f_name in f_in:
            dialog_dirs.append(os.path.join(data_dir, f_name.strip()))

    for i, dialog_dir in enumerate(dialog_dirs):
        dialog = dstc_util.parse_dialog_from_directory(dialog_dir)

        out_dialog = Dialog(dialog_dir, dialog.session_id)
        for utterance in dialog.utterances:
            state = dict(utterance.dialog_state)
            if use_stringified_system_acts:
                msg = _stringify_act(utterance.slots_map)
            else:
                msg = utterance.transcript
            actor_id = Dialog.ACTORS_MAP[utterance.speaker]
            out_dialog.add_message(
                [(msg, 1.0)],
                state,
                actor_id,
                utterance.segment_topic,
                utterance.segment_bio
            )

        with open(os.path.join(out_dir, "%d.json" % (i,)), "w") as f_out:
            f_out.write(out_dialog.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import DSTC5 data to XTrack2."
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help="Root directory with logs."
    )
    parser.add_argument(
        '--flist',
        required=True,
        help="File list with logs."
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="Output directory."
    )
    parser.add_argument(
        '--use_stringified_system_acts',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    ONTOLOGY = build_unified_ontology(
        ontology_reader.OntologyReader(dstc45_ontology_filename)
    )
    import_dstc(**vars(args))
