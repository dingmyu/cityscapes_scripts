#!/bin/sh
exp=res101_psp_coarse_fine
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}
#part=Segmentation
part=Segmentation
numGPU=8
nodeGPU=8
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $part --gres=gpu:$nodeGPU -n$numGPU --ntasks-per-node=$nodeGPU --job-name=${exp} \
python -u train.py \
  --data_root=/mnt/lustre/share/dingmingyu/cityscapes/ \
  --train_list=/mnt/lustre/share/dingmingyu/cityscapes/instance_list_new.txt \
  --val_list=/mnt/lustre/share/dingmingyu/cityscapes/list/fine_val.txt \
  --layers=18 \
  --backbone=resnet \
  --net_type=0 \
  --port=12356 \
  --syncbn=1 \
  --classes=2 \
  --crop_h=233 \
  --crop_w=233 \
  --zoom_factor=4 \
  --base_lr=1e-2 \
  --epochs=200 \
  --start_epoch=1 \
  --batch_size=8 \
  --bn_group=8 \
  --save_step=20 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  --ignore_label 255 \
  --workers 2 \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
