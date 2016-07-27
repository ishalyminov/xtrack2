if [ $# -lt 2 ]; then
    echo "Usage: sh evaluate.sh <challenge_id> <dataset_name>"
    exit
fi

CHALLENGE_ID=$1
DATASET_NAME=$2

python -m dstc5_scripts.check_main \
    --dataset dstc45_dev --ontology dstc5_scripts/config/ontology_dstc5.json \
    --dataroot data/dstc5 --trackfile dstc45_dev.out

python -m dstc5_scripts.score_main \
    --dataset dstc45_dev --dataroot data/dstc5 --trackfile dstc45_dev.out \
    --scorefile dstc45_dev.score \
    --ontology dstc5_scripts/config/ontology_dstc5.json

python -m dstc5_scripts.report_main --scorefile dstc45_dev.score

