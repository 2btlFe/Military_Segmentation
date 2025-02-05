#!/bin/bash

# nclass=9
# img_size=(1024)
# batch_size=(8)
# epochs=(10 20 30)
# lr=(0.0001 0.00005)

# for i in ${img_size[@]}; do
#     for j in ${batch_size[@]}; do
#         for k in ${epochs[@]}; do
#             for l in ${lr[@]}; do
#                 echo "Training UnetFormer with img_size=${i}, batch_size=${j}, epochs=${k}, lr=${l}"
#                 CUDA_VISIBLE_DEVICE=0 python train.py --img_size $i --batch_size $j --num_epochs $k --lr $l
#                 echo "Test UnetFormer with img_size=${i}, batch_size=${j}, epochs=${k}, lr=${l}"
#                 CUDA_VISIBLE_DEVICE=0 python test.py --model_dir "outputs/UnetFormer_img_size_${i}_nclass_${nclass}_lr_${l}_bs_${j}_epochs_${k}"
#             done
#         done
#     done
# done


# nclass=9
# img_size=(768)
# batch_size=(8)
# epochs=(10 20 30)
# lr=(0.0001 0.00005)

nclass=9
img_size=(768)
batch_size=(8)
epochs=(10)
lr=(0.0001)

for i in ${img_size[@]}; do
    for j in ${batch_size[@]}; do
        for k in ${epochs[@]}; do
            for l in ${lr[@]}; do
                # echo "Training UnetFormer with img_size=${i}, batch_size=${j}, epochs=${k}, lr=${l}"
                # CUDA_VISIBLE_DEVICE=0 python train.py --img_size $i --batch_size $j --num_epochs $k --lr $l
                echo "Test UnetFormer with img_size=${i}, batch_size=${j}, epochs=${k}, lr=${l}"
                CUDA_VISIBLE_DEVICE=1 python test.py --model_dir "outputs/UnetFormer_img_size_${i}_nclass_${nclass}_lr_${l}_bs_${j}_epochs_${k}" --fig_dir $1 --skku_dir $2 --sam_mask
            done
        done
    done
done


