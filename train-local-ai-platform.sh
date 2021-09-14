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

set -ev

echo "Training local ML model"



PACKAGE_PATH=trainer

export EPOCHS=1
export DF_PATH=gs://foundation_project2/training_folder/zomato.csv
export JOB_DIR=gs://foundation_project2/bin_files

gcloud ai-platform local train \
        --package-path=${PACKAGE_PATH} \
        --module-name=trainer.train \
        -- \
        --job-dir="${JOB_DIR}" \
        --df-path="$DF_PATH" \
        --epochs=$EPOCHS