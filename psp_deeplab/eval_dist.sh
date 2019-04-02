#!/bin/sh
EXP_DIR=exp/cityscapes/psp50_dist_16_713
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
cp eval.sh eval.py ${EXP_DIR}

srun -p $1 -n$2 --gres=gpu:$3 --ntasks-per-node=$3 \
    --job-name=pred --kill-on-bad-exit=1 \
python eval.py \
  --data_root=/mnt/lustre/zhouhui/cityscape_data/ \
  --val_list=/mnt/lustre/zhuxinge/pspnet/all_txt/city_val_withlabel.txt \
  --split=val \
  --layers=50 \
  --classes=19 \
  --base_size=2048 \
  --crop_h=713 \
  --crop_w=713 \
  --zoom_factor=1 \
  --ignore_label=255 \
  --scales 1.0 \
  --has_prediction=0 \
  --gpu 0 \
  --model_path=${EXP_DIR}/model/train_epoch_$4.pth \
  --save_folder=${EXP_DIR}/result/epoch_$4/val/ss \
  2>&1 | tee ${EXP_DIR}/result/epoch_1-val-ss-$now.log

# --scales 0.5 0.75 1.0 1.25 1.5 1.75 \