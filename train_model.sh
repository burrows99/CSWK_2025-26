#!/bin/bash

# Usage: bash train_model.sh <model_name> [epochs] [batch_size] [learning_rate]
# Example: bash train_model.sh resnet50 15 32 0.0001

MODEL=${1:-tf_efficientnet_b0}
EPOCHS=${2:-10}
BATCH_SIZE=${3:-32}
LR=${4:-0.00005}

echo "Training model: $MODEL"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Learning Rate: $LR"

STUDENT_ID=6915661 STUDENT_NAME="Raunak Burrows" python Training.py \
--model_mode $MODEL \
--dataset_location ../EEEM066_KnifeHunter \
--train_datacsv dataset/train.csv \
--val_datacsv dataset/validation.csv \
--saved_checkpoint_path Knife-$MODEL \
--epochs $EPOCHS \
--batch_size $BATCH_SIZE \
--n_classes 543 \
--learning_rate $LR \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 0 \
--brightness 0.2 \
--contrast 0.2 \
--saturation 0.2 \
--hue 0.2 \
--optim adam \
--lr-scheduler CosineAnnealingLR
