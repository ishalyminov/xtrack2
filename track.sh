#!/usr/bin/env sh

for topic in `python -m ontology_util get_topics`; do
    echo "Tracking dialogs on '$topic' topic"
    python -m dstc_tracker \
        --dataset_name dstc45_dev \
        --data_file data/xtrack/e2_tagged_${topic}/dev.json \
        --output_file dstc5_dev_${topic}.out \
        --params_file xtrack_oute2_tagged_${topic}/params.final.p \
        --ontology dstc5_scripts/config/ontology_dstc5.json
done

python -m merge_topic_results \
    --dialogs_folder data/dstc5 \
    --result_trackfile dstc45_dev.out
