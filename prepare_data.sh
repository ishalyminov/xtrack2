#!/usr/bin/env sh

python -m translate_dialogs --input_dir data/dstc5 --output_dir data/dstc5_translated
python -m chop_dstc45_dialogs --dialogs_folder data/dstc5_translated
python -m rearrange_datasets --dialogs_folder data/dstc5_chopped
python -m build_data_dstc45