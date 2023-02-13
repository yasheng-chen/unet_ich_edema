#!/bin/bash

gpu_id=0

model_id=72

for group_id in {1..10} 
do
   name_dir_output=./Results_model_${model_id}/group_${group_id}
   mkdir $name_dir_output

   python main.py --network=unet --gpu_id=${gpu_id} --model_in_path=./models/group_${group_id}/model_${model_id}/model_${model_id}-${model_id} --data_in=./normalized --output_dir=${name_dir_output} --manual_splits=./manual_splits.txt
done

model_id=114

for group_id in {1..10} 
do
   name_dir_output=./Results_model_${model_id}/group_${group_id}
   mkdir $name_dir_output

   python main.py --network=unet --gpu_id=${gpu_id} --model_in_path=./models/group_${group_id}/model_${model_id}/model_${model_id}-${model_id} --data_in=./normalized --output_dir=${name_dir_output} --manual_splits=manual_splits.txt
done

####ich
python ensemble_outputs.py --path ./Results_model_114/group_1 --class_id 1 --ngroups 10

####edema
python ensemble_outputs.py --path ./Results_model_72/group_1 --class_id 2 --ngroups 10


