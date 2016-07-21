#!/usr/bin/env sh

python translate_dialogs.py --input_dir data/dstc5 --output_dir data/dstc5_translated
python chop_dstc45_dialogs.py --dialogs_folder data/dstc5_translated
python build_data_dstc45.py