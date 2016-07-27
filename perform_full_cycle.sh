#!/usr/bin/env sh

. prepare_data.sh

. train_trackers.sh


exit

python dstc_tracker.py \
    --dataset_name dstc5_dev \
    --data_file data/xtrack/e2_tagged_xtrack_dstc45/dev.json \
    --output_file dstc5_dev.out \
    --params_file xtrack_out/params.final.p \
    --ontology dstc5_scripts/config/ontology_dstc5.json

python dstc5_scripts/check_main.py \
    --dataset dstc5_dev --ontology dstc5_scripts/config/ontology_dstc5.json \
    --dataroot data/dstc5 --trackfile dstc5_dev.out

python dstc5_scripts/score_main.py \
    --dataset dstc5_dev --dataroot data/dstc5 --trackfile dstc5_dev.out
    --scorefile dstc5_dev.score \
    --ontology dstc5_scripts/config/ontology_dstc5.json

python dstc5_scripts/report_main.py --scorefile dstc5_dev.score
