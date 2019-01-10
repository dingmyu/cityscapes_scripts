dir_name = '/mnt/lustre/share/dingmingyu/cityscapes/'
ignore = '/mnt/lustre/share/ADAS_DATA/ignore.png'
all_list = []

for i in range(100):
	all_list.append(dir_name + 'result_person_new_val/%d_image.png' % i)
for i in range(100):
	all_list.append(dir_name + 'result_car_new_val/%d_image.png' % i)

import random
import cv2
random.shuffle(all_list)

for item in all_list:
	if cv2.imread(item) is not None:
		if 'person' in item:
			print item, item.replace('image', 'label'), ignore
		if 'car' in item:
			print item, ignore, item.replace('image', 'label')
