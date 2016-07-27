#!/usr/bin/env sh

for topic in `python -m ontology_util get_topics`; do
    echo "Training the tracker for '$topic' topic"
    python -m xtrack2 \
        --mb_size=16 --lr=0.1 --opt_type=rmsprop --p_drop=0.01 --n_cells=100 \
        --n_epochs=20 --ontology dstc5_scripts/config/ontology_dstc5.json \
        data/xtrack/e2_tagged_$topic
done
