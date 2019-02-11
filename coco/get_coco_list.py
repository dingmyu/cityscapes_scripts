#dir_name = '/mnt/lustre/share/DSK/datasets/mscoco2017/train2017/'
dir_name = '/mnt/lustre/dingmingyu/workspace/instance_seg/data/'
ignore = '/mnt/lustre/share/ADAS_DATA/ignore.png'
all_list = []
import os

car_list = os.listdir('car_ori')
person_list = os.listdir('person_ori')
for i in car_list:
	all_list.append(dir_name + 'car_ori/%s' % i)
for i in person_list:
	all_list.append(dir_name + 'person_ori/%s' % i)

import random
import cv2
random.shuffle(all_list)

for item in all_list:
	if cv2.imread(item) is not None:
		if 'person' in item:
			print item, item.replace('ori', 'gt'), ignore
		if 'car' in item:
			print item, ignore, item.replace('`ori', 'gt')
