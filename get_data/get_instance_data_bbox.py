import glob
import os
import json
import PIL.Image     as Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import cv2
bbox_dict = {}
f  = open('/mnt/lustre/share/xyzeng/public/cityscapes/results_val.txt').readlines()
for index,line in enumerate(f):
    #if index> 100:
    #    break
    line = line.strip().split()
    image = line[0]#.replace('/','')[:-4]
    classes = line[1]
    bbox = [float(item) for item in line[-4:]]
    bbox_dict[image] = bbox_dict.get(image, [])
    bbox_dict[image].append(bbox)

def iou(arr1, arr2):
    u = (arr1[2]-arr1[0])*(arr1[3]-arr1[1]) + (arr2[2]-arr2[0])*(arr2[3]-arr2[1])
    i = max(min(arr1[2],arr2[2])-max(arr1[0],arr2[0]), 0) *  max(min(arr1[3],arr2[3])-max(arr1[1],arr2[1]), 0)
    #print(u,i,max(min(arr1[2],arr2[2]-max(arr1[0],arr2[0])), 0))
    return float(i)/(u-i)



filelist = open('list/fine_val.txt').readlines()
num_car = 0
num_person = 0
for index, line in enumerate(filelist):
	print(index)
	image, label = line.strip().split()
	print image[12:]
	label = label.replace('_labelTrainIds.png','_polygons.json')
	json_file = json.load(open(label))
	for obj in json_file['objects']:
		if obj['label'] == 'car' and image[12:] in bbox_dict:
			car = obj
			for bbox in bbox_dict[image[12:]]:
				if (iou(list(np.array(car['polygon']).min(0)) + (list(np.array(car['polygon']).max(0))), bbox)) > 0.8:
	#			if (np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)).max() > 60:
					instanceImg = Image.new("1", tuple(np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)))
					drawer = ImageDraw.Draw( instanceImg )
					polygon = [(item[0]-np.array(car['polygon']).min(0)[0],item[1]-np.array(car['polygon']).min(0)[1]) for item in car['polygon']]
					drawer.polygon(polygon, fill=1)
					instanceImg.save('result_car_new_val/%d_label.png' % num_car)
					picture  = cv2.imread(image)
					cv2.imwrite('result_car_new_val/%d_image.png' % num_car, picture[np.array(car['polygon']).min(0)[1]:np.array(car['polygon']).max(0)[1],np.array(car['polygon']).min(0)[0]:np.array(car['polygon']).max(0)[0]])
					num_car += 1
					break
		if obj['label'] == 'person' and image[12:] in bbox_dict:
			car = obj
			for bbox in bbox_dict[image[12:]]:
				if (iou(list(np.array(car['polygon']).min(0)) + (list(np.array(car['polygon']).max(0))), bbox)) > 0.8:
			#if (np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)).max() > 40:
					instanceImg = Image.new("1", tuple(np.array(car['polygon']).max(0)-np.array(car['polygon']).min(0)))
					drawer = ImageDraw.Draw( instanceImg )
					polygon = [(item[0]-np.array(car['polygon']).min(0)[0],item[1]-np.array(car['polygon']).min(0)[1]) for item in car['polygon']]
					drawer.polygon(polygon, fill=1)
					instanceImg.save('result_person_new_val/%d_label.png' % num_person)
					picture  = cv2.imread(image)
					cv2.imwrite('result_person_new_val/%d_image.png' % num_person, picture[np.array(car['polygon']).min(0)[1]:np.array(car['polygon']).max(0)[1],np.array(car['polygon']).min(0)[0]:np.array(car['polygon']).max(0)[0]])
					num_person += 1
					break

