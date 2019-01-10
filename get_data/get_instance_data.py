import glob
import os
import json
import PIL.Image     as Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import cv2
filelist = open('list/fine_train.txt').readlines()
num_car = 0
num_person = 0
for index, line in enumerate(filelist):
	print(index)
	image, label = line.strip().split()
	label = label.replace('_labelTrainIds.png','_polygons.json')
	json_file = json.load(open(label))
	for obj in json_file['objects']:
		if obj['label'] == 'car':
			car = obj
			if (np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)).max() > 60:
				instanceImg = Image.new("1", tuple(np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)))
				drawer = ImageDraw.Draw( instanceImg )
				polygon = [(item[0]-np.array(car['polygon']).min(0)[0],item[1]-np.array(car['polygon']).min(0)[1]) for item in car['polygon']]
				drawer.polygon(polygon, fill=1)
				instanceImg.save('result_car/%d_label.png' % num_car)
				picture  = cv2.imread(image)
				cv2.imwrite('result_car/%d_image.png' % num_car, picture[np.array(car['polygon']).min(0)[1]:np.array(car['polygon']).max(0)[1],np.array(car['polygon']).min(0)[0]:np.array(car['polygon']).max(0)[0]])
				num_car += 1
		if obj['label'] == 'person':
			car = obj
			if (np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)).max() > 40:
				instanceImg = Image.new("1", tuple(np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)))
				drawer = ImageDraw.Draw( instanceImg )
				polygon = [(item[0]-np.array(car['polygon']).min(0)[0],item[1]-np.array(car['polygon']).min(0)[1]) for item in car['polygon']]
				drawer.polygon(polygon, fill=1)
				instanceImg.save('result_person/%d_label.png' % num_person)
				picture  = cv2.imread(image)
				cv2.imwrite('result_person/%d_image.png' % num_person, picture[np.array(car['polygon']).min(0)[1]:np.array(car['polygon']).max(0)[1],np.array(car['polygon']).min(0)[0]:np.array(car['polygon']).max(0)[0]])
				num_person += 1

