#!/bin/sh
EXP_DIR=exp/cityscapes/psp50_dist_16_713_fine_4
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
cp eval_multiscale.sh eval.py ${EXP_DIR}

srun -p $1 -n$2 --gres=gpu:$3 --ntasks-per-node=$3 \
    --job-name=pred --kill-on-bad-exit=1 \
python -u eval.py \
  --data_root=/mnt/lustre/share/dingmingyu/cityscapes/ \
  --val_list=/mnt/lustre/share/dingmingyu/cityscapes/list/fine_test.txt \
  --split=val \
  --layers=101 \
  --classes=19 \
  --base_size=2048 \
  --crop_h=713 \
  --crop_w=713 \
  --zoom_factor=2 \
  --ignore_label=255 \
  --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
  --has_prediction=0 \
  --gpu 0 \
  --model_path=${EXP_DIR}/model/train_epoch_$4.pth \
  --save_folder=${EXP_DIR}/result/epoch_$4/val/ss \
  2>&1 | tee ${EXP_DIR}/result/epoch_$4-val-ss-multiscale-$now.log

# --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
