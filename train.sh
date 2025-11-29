#!/bin/bash

STUDENT_ID=6915661 STUDENT_NAME="Raunak Burrows" python Training.py \
--model_mode tf_efficientnet_b0 \
--dataset_location ../EEEM066_KnifeHunter \
--train_datacsv dataset/train.csv \
--val_datacsv dataset/validation.csv \
--saved_checkpoint_path Knife-Effb0 \
--epochs 10 \
--batch_size 32 \
--n_classes 543 \
--learning_rate 0.00005 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 0 \
--brightness 0.2 \
--contrast 0.2 \
--saturation 0.2 \
--hue 0.2 \
--optim adam \
--lr-scheduler CosineAnnealingLR
