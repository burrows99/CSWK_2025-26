#!/bin/bash

STUDENT_ID=6915661 STUDENT_NAME="Raunak Burrows" python Testing.py \
--model_mode tf_efficientnet_b0 \
--model-path Knife-Effb0/Knife-tf_efficientnet_b0-E10.pth \
--dataset_location ../EEEM066_KnifeHunter \
--test_datacsv dataset/test.csv \
--seed 0 \
--batch_size 32 \
--n_classes 543 \
--resized_img_weight 224 \
--resized_img_height 224 \
--evaluate-only 
