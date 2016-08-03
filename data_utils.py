from data import *
from data_baseline import *
import import_dstc45
from xtrack2_config import (
    experiment_directory,
    dstc45_config_dir
)


def load_dialogs(data_dir):
    dialogs = []
    for f_name in sorted(os.listdir(data_dir), key=lambda x: x.split('.')[0]):
        if f_name.endswith('.json'):
            dialog = data_model.Dialog.deserialize(
                open(os.path.join(data_dir, f_name)).read()
            )
            dialogs.append(dialog)
    return dialogs


def parse_slots_and_slot_groups(args):
    slot_groups = {}
    slots = []
    for i, slot_group in enumerate(args.slots.split(':')):
        if '=' in slot_group:
            name, vals = slot_group.split('=', 1)
        else:
            name = 'grp%d' % i
            vals = slot_group
        slot_group = vals.split(',')
        slot_groups[name] = slot_group
        for slot in slot_group:
            if not slot in slots:
                slots.append(slot)
    return slot_groups, slots


def import_dstc_data(in_data_directory, out_dir, dataset):
    input_dir = in_data_directory
    flist = os.path.join(dstc45_config_dir, '{}.flist'.format(dataset))
    import_dstc45.import_dstc(
        data_dir=input_dir,
        out_dir=out_dir,
        flist=flist,
        use_stringified_system_acts=False
    )
    return out_dir


def prepare_experiment(
    experiment_name,
    data_directory,
    slots,
    slot_groups,
    ontology,
    skip_dstc_import_step,
    builder_opts,
    builder_type,
    in_datasets
):
    e_root = os.path.join(experiment_directory, 'xtrack/%s' % experiment_name)
    debug_dir = os.path.join(e_root, 'debug')

    based_on = None
    for src_dataset, dst_dataset in in_datasets:
        out_dir = os.path.join(e_root, src_dataset)
        if not skip_dstc_import_step:
            import_dstc_data(data_directory, out_dir, src_dataset)
        dialogs = load_dialogs(out_dir)

        logging.info('Initializing.')
        if builder_type == 'baseline':
            builder_cls = DataBuilderBaseline
        elif builder_type == 'xtrack_dstc45':
            builder_cls = DataBuilder
        else:
            raise Exception('unknown builder')

        xtd_builder = builder_cls(
            based_on=based_on,
            include_base_seqs=False,
            slots=slots,
            slot_groups=slot_groups,
            oov_ins_p=0.1 if dst_dataset == 'train' else 0.0,
            word_drop_p=0.0,
            include_system_utterances=True,
            nth_best=0,
            ontology=ontology,
            debug_dir=debug_dir,
            **builder_opts
        )
        logging.info('Building.')
        xtd = xtd_builder.build(dialogs)

        logging.info('Saving.')
        out_file = os.path.join(e_root, '%s.json' % dst_dataset)
        xtd.save(out_file)

        if dst_dataset == 'train':
            based_on = out_file
