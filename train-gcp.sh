#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs local training for a TensorFlow model.
# Link: https://cloud.google.com/ai-platform/docs/getting-started-keras

set -ev

echo "Training local ML model"



PACKAGE_PATH=trainer

export EPOCHS=50
export DF_PATH=gs://foundation_project2/training_folder/zomato.csv
JOBDIR=gs://foundation_project2/bin_files

REGION="us-central1" # choose a gcp region from https://cloud.google.com/ml engine/docs/tensorflow/regions
TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
MODEL_NAME="foodrecommender" # change to your model name

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
gcloud ai-platform jobs submit training ${JOB_NAME} \
        --python-version=3.7 \
        --runtime-version=2.3 \
        --region=${REGION} \
        --job-dir=$JOBDIR \
        --scale-tier=${TIER} \
        --module-name=trainer.train \
        --package-path=${PACKAGE_PATH}  \
        -- \
        --df-path="$DF_PATH" \
        --epochs=$EPOCHS
		#--config hptuning_config.yaml \