#!/usr/bin/env sh

if [ $# -lt 2 ]; then
    echo "Usage: sh track.sh <challenge_id> <dataset_name>"
    exit
fi

CHALLENGE_ID=$1
DATASET_NAME=$2

for topic in `python -m ontology_util get_topics`; do
    echo "Tracking dialogs on '$topic' topic"
    python -m dstc_tracker \
        --dataset "${CHALLENGE_ID}_${DATASET_NAME}" \
        --data_file data/xtrack/e2_tagged_${CHALLENGE_ID}_${topic}/${DATASET_NAME}.json \
        --output_file ${CHALLENGE_ID}_${DATASET_NAME}_${topic}.out \
        --params_file xtrack2_oute2_tagged_${topic}/params.final.p \
        --ontology dstc5_scripts/config/ontology_dstc5.json
done

python -m merge_topic_results \
    --dialogs_folder data/dstc5_translated \
    --chopped_dialogs_folder data/dstc5_chopped_${DATASET_NAME} \
    --dataset_name "${CHALLENGE_ID}_${DATASET_NAME}" \
    --trackfiles_to_merge "${CHALLENGE_ID}_${DATASET_NAME}_FOOD.out,${CHALLENGE_ID}_${DATASET_NAME}_ATTRACTION.out,${CHALLENGE_ID}_${DATASET_NAME}_TRANSPORTATION.out,${CHALLENGE_ID}_${DATASET_NAME}_SHOPPING.out,${CHALLENGE_ID}_${DATASET_NAME}_ACCOMMODATION.out" \
    --result_trackfile ${CHALLENGE_ID}_${DATASET_NAME}.out
