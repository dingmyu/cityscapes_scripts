#!/bin/sh
EXP_DIR=exp/cityscapes/psp50_dist_16_713
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}

srun -p $1 -n$2 --gres=gpu:$3 --ntasks-per-node=$3 \
    --job-name=pred --kill-on-bad-exit=1 \
python -u train.py \
  --data_root=/mnt/lustre/share/dingmingyu/cityscapes/ \
  --train_list=/mnt/lustre/share/dingmingyu/cityscapes/list/coarse_train_shuffle.txt \
  --val_list=/mnt/lustre/share/dingmingyu/cityscapes/list/fine_val.txt \
  --layers=101 \
  --bn_group=$2 \
  --dist=1 \
  --port=12345 \
  --syncbn=1 \
  --classes=19 \
  --crop_h=713 \
  --crop_w=713 \
  --zoom_factor=2 \
  --gpu 0 1 2 3 4 5 6 7\
  --base_lr=1e-2 \
  --epochs=120 \
  --save_step=20 \
  --start_epoch=1 \
  --batch_size=2 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
