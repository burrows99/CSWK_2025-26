#!/bin/bash

# Usage: bash test_model.sh <model_name> [epochs] [batch_size]
# Example: bash test_model.sh resnet50 15 32

MODEL=${1:-tf_efficientnet_b0}
EPOCHS=${2:-10}
BATCH_SIZE=${3:-32}

echo "Testing model: $MODEL"
echo "Model path: Knife-$MODEL/Knife-$MODEL-E$EPOCHS.pth"

STUDENT_ID=6915661 STUDENT_NAME="Raunak Burrows" python Testing.py \
--model_mode $MODEL \
--model-path Knife-$MODEL/Knife-$MODEL-E$EPOCHS.pth \
--dataset_location ../EEEM066_KnifeHunter \
--test_datacsv dataset/test.csv \
--seed 0 \
--batch_size $BATCH_SIZE \
--n_classes 543 \
--resized_img_weight 224 \
--resized_img_height 224 \
--evaluate-only
